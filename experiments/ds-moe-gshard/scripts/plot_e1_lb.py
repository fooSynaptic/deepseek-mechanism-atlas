#!/usr/bin/env python3
"""E1 after LB fix: val BPB + aux/CE + max_expert_frac."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "e1_lb_archive"
OUT = ROOT / "docs" / "figures"

RUNS = {
    "GShard": ART / "e1_lb_gshard_w20" / "train.jsonl",
    "DeepSeekMoE": ART / "e1_lb_dsmoe" / "train.jsonl",
}
COLORS = {"GShard": "#2563eb", "DeepSeekMoE": "#dc2626"}


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
    """Markdown preview often fails on matplotlib SVG DOCTYPE / RDF."""
    text = path.read_text()
    if text.startswith("<?xml"):
        text = text.split(">", 1)[1].lstrip()
    if text.startswith("<!DOCTYPE"):
        text = text.split(">", 1)[1].lstrip()
    path.write_text(text)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {k: load(p) for k, p in RUNS.items()}
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "figure.facecolor": "#f8fafc",
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), constrained_layout=True)
    fig.suptitle("E1 after load-balance fix — 10B, same data_l2", fontsize=13)

    ax = axes[0, 0]
    for name, d in data.items():
        xs = [r["tokens"] / 1e9 for r in d["eval"]]
        ys = [r["val_bpb"] for r in d["eval"]]
        ax.plot(xs, ys, "o-", color=COLORS[name], lw=1.6, ms=4.5, label=name)
    ax.set_title("Val BPB (primary)")
    ax.set_xlabel("tokens seen (B)")
    ax.set_ylabel("val BPB (lower is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_xlim(0, 10.5)

    ax = axes[0, 1]
    for name, d in data.items():
        xs = [r["tokens"] / 1e9 for r in d["train"] if "ce" in r]
        ys = [float(r["ce"]) for r in d["train"] if "ce" in r]
        sx, sy = smooth(xs, ys, 9)
        ax.plot(sx, sy, color=COLORS[name], lw=1.6, label=name)
    ax.set_title("Train CE (smoothed)")
    ax.set_xlabel("tokens seen (B)")
    ax.set_ylabel("cross-entropy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_xlim(0, 10.5)

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
        ax.plot(sx, sy, color=COLORS[name], lw=1.6, label=name)
    ax.axhline(1.0, color="#16a34a", ls="--", lw=1, label="1% floor")
    ax.set_title("aux / CE (%)")
    ax.set_xlabel("tokens seen (B)")
    ax.set_ylabel("percent")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 10.5)

    ax = axes[1, 1]
    for name, d in data.items():
        xs = [r["tokens"] / 1e9 for r in d["train"]]
        ys = [float(r["max_expert_frac"]) for r in d["train"]]
        sx, sy = smooth(xs, ys, 9)
        ax.plot(sx, sy, color=COLORS[name], lw=1.6, label=name)
    ax.axhline(2 / 16, color="#64748b", ls=":", lw=1, label="uniform GShard k/N")
    ax.axhline(7 / 63, color="#a855f7", ls=":", lw=1, label="uniform DSMoE k/N")
    ax.axhline(0.5, color="#f59e0b", ls="--", lw=1, label="C2 collapse line")
    ax.set_title("max_expert_frac")
    ax.set_xlabel("tokens seen (B)")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, 10.5)

    for ax in axes.flat:
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    svg = OUT / "e1_lb_curves.svg"
    png = OUT / "e1_lb_curves.png"
    fig.savefig(svg, format="svg", bbox_inches="tight")
    fig.savefig(png, format="png", dpi=160, bbox_inches="tight")
    _strip_svg_dtd(svg)
    print(f"wrote {svg}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
