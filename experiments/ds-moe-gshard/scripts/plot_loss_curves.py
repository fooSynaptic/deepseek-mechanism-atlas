#!/usr/bin/env python3
"""Plot train CE loss and val BPB curves from train.jsonl artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
OUT = ROOT / "docs" / "figures"

RUNS = {
    "E1 GShard (w20)": ART / "e1_archive" / "e1_gshard_w20" / "train.jsonl",
    "E1 DSMoE (w20)": ART / "e1_archive" / "e1_dsmoe" / "train.jsonl",
    "E2 GShard (w20)": ART / "e2_archive" / "e2_gshard_w20" / "train.jsonl",
    "E2 DSMoE": ART / "e2_archive" / "e2_dsmoe" / "train.jsonl",
}

COLORS = {
    "E1 GShard (w20)": "#2563eb",
    "E1 DSMoE (w20)": "#dc2626",
    "E2 GShard (w20)": "#60a5fa",
    "E2 DSMoE": "#f87171",
}


def load_series(path: Path) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return tokens_B, ce, val_tokens_B, val_bpb."""
    tok_b: list[float] = []
    ce: list[float] = []
    val_tok_b: list[float] = []
    val_bpb: list[float] = []
    if not path.exists():
        return tok_b, ce, val_tok_b, val_bpb
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            o = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "ce" in o and "tokens" in o:
            tok_b.append(o["tokens"] / 1e9)
            ce.append(float(o["ce"]))
        if "val_bpb" in o and "tokens" in o:
            val_tok_b.append(o["tokens"] / 1e9)
            val_bpb.append(float(o["val_bpb"]))
    return tok_b, ce, val_tok_b, val_bpb


def smooth(xs: list[float], ys: list[float], window: int = 5) -> tuple[list[float], list[float]]:
    if len(ys) < window:
        return xs, ys
    out_y: list[float] = []
    out_x: list[float] = []
    half = window // 2
    for i in range(half, len(ys) - half):
        chunk = ys[i - half : i + half + 1]
        out_y.append(sum(chunk) / len(chunk))
        out_x.append(xs[i])
    return out_x, out_y


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "figure.facecolor": "#f8fafc",
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    fig.suptitle("DeepSeekMoE vs GShard — loss convergence (data_l2)", fontsize=13, y=1.01)

    # Panel A: E1 train CE
    ax = axes[0, 0]
    for label in ("E1 GShard (w20)", "E1 DSMoE (w20)"):
        x, y, _, _ = load_series(RUNS[label])
        if not x:
            continue
        sx, sy = smooth(x, y, window=9)
        ax.plot(sx, sy, label=label, color=COLORS[label], linewidth=1.8)
    ax.set_title("E1 (10B) — train CE (smoothed)")
    ax.set_xlabel("tokens seen (B)")
    ax.set_ylabel("cross-entropy loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_xlim(0, 10.5)

    # Panel B: E1 val BPB
    ax = axes[0, 1]
    for label in ("E1 GShard (w20)", "E1 DSMoE (w20)"):
        _, _, vx, vy = load_series(RUNS[label])
        if not vx:
            continue
        ax.plot(vx, vy, "o-", label=label, color=COLORS[label], linewidth=1.5, markersize=4)
    ax.set_title("E1 (10B) — val BPB")
    ax.set_xlabel("tokens seen (B)")
    ax.set_ylabel("val BPB (lower is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_xlim(0, 10.5)

    # Panel C: E2 train CE
    ax = axes[1, 0]
    for label in ("E2 GShard (w20)", "E2 DSMoE"):
        x, y, _, _ = load_series(RUNS[label])
        if not x:
            continue
        sx, sy = smooth(x, y, window=15)
        ax.plot(sx, sy, label=label, color=COLORS[label], linewidth=1.8)
    ax.set_title("E2 (40B) — train CE (smoothed)")
    ax.set_xlabel("tokens seen (B)")
    ax.set_ylabel("cross-entropy loss")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_xlim(10, 40.5)
    ax.axvline(10, color="#94a3b8", linestyle="--", linewidth=0.8, alpha=0.7)

    # Panel D: E2 val BPB
    ax = axes[1, 1]
    for label in ("E2 GShard (w20)", "E2 DSMoE"):
        _, _, vx, vy = load_series(RUNS[label])
        if not vx:
            continue
        suffix = "" if label == "E2 GShard (w20)" else " (partial)" if max(vx) < 39.5 else ""
        ax.plot(vx, vy, "o-", label=label + suffix, color=COLORS[label], linewidth=1.5, markersize=3.5)
    ax.set_title("E2 (40B) — val BPB")
    ax.set_xlabel("tokens seen (B)")
    ax.set_ylabel("val BPB (lower is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_xlim(10, 40.5)
    ax.axvline(10, color="#94a3b8", linestyle="--", linewidth=0.8, alpha=0.7)

    for ax in axes.flat:
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))

    svg_path = OUT / "loss_curves.svg"
    png_path = OUT / "loss_curves.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=160, bbox_inches="tight")
    print(f"wrote {svg_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
