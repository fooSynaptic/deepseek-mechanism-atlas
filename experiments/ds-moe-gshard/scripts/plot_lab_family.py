#!/usr/bin/env python3
"""Lab family curves: GShard, DSMoE, GShardx1.5, F-train, S-train."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

RUNS = {
    "GShard Top-2": ROOT / "artifacts" / "e1_lb_archive" / "e1_lb_gshard_w20" / "train.jsonl",
    "DeepSeekMoE 1+7": ROOT / "artifacts" / "e1_lb_archive" / "e1_lb_dsmoe" / "train.jsonl",
    "GShard x1.5": ROOT / "artifacts" / "e3_archive" / "train.jsonl",
    "F-train 1+3": ROOT / "artifacts" / "f_train_archive" / "train.jsonl",
    "S-train 0+8": ROOT / "artifacts" / "s_train_archive" / "train.jsonl",
}
COLORS = {
    "GShard Top-2": "#2563eb",
    "DeepSeekMoE 1+7": "#dc2626",
    "GShard x1.5": "#059669",
    "F-train 1+3": "#d97706",
    "S-train 0+8": "#7c3aed",
}
STYLES = {
    "GShard Top-2": "-",
    "DeepSeekMoE 1+7": "-",
    "GShard x1.5": "--",
    "F-train 1+3": "-.",
    "S-train 0+8": ":",
}


def load(path: Path) -> dict:
    train: list[dict] = []
    evals: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        o = json.loads(line)
        if "val_bpb" in o:
            evals.append(o)
        elif "ce" in o:
            train.append(o)
    return {"train": train, "eval": evals}


def smooth(xs: list[float], ys: list[float], window: int = 9) -> tuple[list[float], list[float]]:
    if len(ys) < window:
        return xs, ys
    half = window // 2
    out_x, out_y = [], []
    for i in range(half, len(ys) - half):
        chunk = ys[i - half : i + half + 1]
        out_y.append(sum(chunk) / len(chunk))
        out_x.append(xs[i])
    return out_x, out_y


def _strip_svg_dtd(path: Path) -> None:
    text = path.read_text()
    if text.startswith("<?xml"):
        text = text.split(">", 1)[1].lstrip()
    if text.startswith("<!DOCTYPE"):
        text = text.split(">", 1)[1].lstrip()
    path.write_text(text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {k: load(p) for k, p in RUNS.items() if p.exists()}
    missing = [k for k, p in RUNS.items() if not p.exists()]
    if missing:
        raise SystemExit(f"missing train.jsonl for: {missing}")

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 8,
            "figure.facecolor": "#f8fafc",
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)
    fig.suptitle(
        "Lab family @ 10B — GShard / DSMoE / x1.5 / F-train / S-train",
        fontsize=13,
    )

    ax = axes[0, 0]
    for name, d in data.items():
        xs = [r["tokens"] / 1e9 for r in d["eval"]]
        ys = [r["val_bpb"] for r in d["eval"]]
        ax.plot(
            xs,
            ys,
            color=COLORS[name],
            linestyle=STYLES[name],
            marker="o",
            lw=1.6,
            ms=4,
            label=name,
        )
    ax.axhline(0.7970, color="#2563eb", ls=":", lw=0.9, alpha=0.7)
    ax.axhline(0.7850, color="#dc2626", ls=":", lw=0.9, alpha=0.7)
    ax.set_title("Val BPB (primary)")
    ax.set_xlabel("tokens seen (B)")
    ax.set_ylabel("val BPB (lower is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0.75, 2.0)

    ax = axes[0, 1]
    # zoom late train
    for name, d in data.items():
        xs = [r["tokens"] / 1e9 for r in d["eval"]]
        ys = [r["val_bpb"] for r in d["eval"]]
        ax.plot(
            xs,
            ys,
            color=COLORS[name],
            linestyle=STYLES[name],
            marker="o",
            lw=1.6,
            ms=4.5,
            label=name,
        )
    ax.set_title("Val BPB — late zoom (4B–10B)")
    ax.set_xlabel("tokens seen (B)")
    ax.set_ylabel("val BPB")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_xlim(4, 10.5)
    ax.set_ylim(0.78, 1.0)

    ax = axes[1, 0]
    for name, d in data.items():
        xs, ys = [], []
        for r in d["train"]:
            ce = float(r.get("ce") or 0.0)
            if ce <= 0:
                continue
            xs.append(r["tokens"] / 1e9)
            ys.append(100.0 * float(r["aux"]) / ce)
        sx, sy = smooth(xs, ys, 9)
        ax.plot(sx, sy, color=COLORS[name], ls=STYLES[name], lw=1.5, label=name)
    ax.axhline(1.0, color="#16a34a", ls="--", lw=1, label="1% floor")
    ax.set_title("aux / CE (%)")
    ax.set_xlabel("tokens seen (B)")
    ax.set_ylabel("percent")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", ncol=2)
    ax.set_xlim(0, 10.5)

    ax = axes[1, 1]
    for name, d in data.items():
        xs = [r["tokens"] / 1e9 for r in d["train"]]
        ys = [float(r["max_expert_frac"]) for r in d["train"]]
        sx, sy = smooth(xs, ys, 9)
        ax.plot(sx, sy, color=COLORS[name], ls=STYLES[name], lw=1.5, label=name)
    ax.axhline(0.5, color="#f59e0b", ls="--", lw=1, label="collapse line")
    ax.set_title("max_expert_frac")
    ax.set_xlabel("tokens seen (B)")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=2)
    ax.set_xlim(0, 10.5)

    for ax in axes.flat:
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    svg = OUT / "lab_family_curves.svg"
    png = OUT / "lab_family_curves.png"
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(png, format="png", dpi=160, bbox_inches="tight")
    _strip_svg_dtd(svg)
    print(f"wrote {svg}")
    print(f"wrote {png}")

    # delta bar chart
    fig2, ax = plt.subplots(figsize=(8.2, 3.8), constrained_layout=True)
    labels = [
        "DSMoE 1+7\nvs GShard",
        "GShard x1.5\nvs GShard",
        "F-train 1+3\nvs GShard",
        "S-train 0+8\nvs DSMoE 1+7",
        "S-train 0+8\nvs GShard",
    ]
    vals = [-0.0120, -0.0110, +0.0044, +0.0083, -0.0037]
    colors = ["#dc2626", "#059669", "#d97706", "#7c3aed", "#9333ea"]
    bars = ax.bar(labels, vals, color=colors, width=0.62)
    ax.axhline(0, color="#334155", lw=1)
    ax.axhline(0.015, color="#94a3b8", ls=":", lw=1, label="~last-1B scatter")
    ax.axhline(-0.015, color="#94a3b8", ls=":", lw=1)
    ax.set_ylabel("Δ val BPB (negative = better than ref)")
    ax.set_title("Final gaps @ 10B — S-train soft vs DSMoE; surgery S-eval is separate")
    ax.grid(True, axis="y", alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v + (0.0015 if v >= 0 else -0.0035),
            f"{v:+.4f}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=8,
            fontweight="bold",
        )
    ax.legend(loc="lower left", fontsize=8)
    png2 = OUT / "lab_final_gaps.png"
    svg2 = OUT / "lab_final_gaps.svg"
    fig2.savefig(png2, format="png", dpi=160, bbox_inches="tight")
    fig2.savefig(svg2, format="svg", bbox_inches="tight")
    _strip_svg_dtd(svg2)
    print(f"wrote {png2}")
    print(f"wrote {svg2}")


if __name__ == "__main__":
    main()
