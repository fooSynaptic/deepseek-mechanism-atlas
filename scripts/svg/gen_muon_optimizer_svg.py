#!/usr/bin/env python3
"""Muon optimizer: AdamW vs Muon split + Algorithm 1 step pipeline (V4)."""
from __future__ import annotations

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DIAGRAMS, FIGURES  # noqa: E402

from gen_deepseek_svgs import box, line, phase, txt, write  # noqa: E402

OUT = str(DIAGRAMS)
FIG_V4 = str(FIGURES / "v4")
PF = "muo"

W, H = 920, 530
MID = W // 2


def _node(cx: int, cy: int, w: int, h: int, title: str, lines: list[str], fill: str, stroke: str) -> str:
    x, y = cx - w // 2, cy - h // 2
    s = box(x, y, w, h, fill, stroke)
    s += txt(cx, y + 18, title)
    for i, ln in enumerate(lines):
        s += txt(cx, y + 34 + i * 14, ln, "dt")
    return s


def gen_muon_optimizer() -> None:
    s = ""
    s += txt(MID, 22, "DeepSeek-V4 Muon optimizer", "t")
    s += txt(MID, 40, "matrix orthogonal update (most params) vs AdamW (embed / head / RMSNorm / mHC gates)", "st")

    # --- top: parameter split ---
    y0, ph_h = 58, 128
    s += phase(24, y0, 872, ph_h, "Parameter split (Basic Configurations, paper sec 2.4)", "#4A90D9")
    cy = y0 + 66
    s += _node(
        230,
        cy,
        360,
        84,
        "AdamW",
        ["embed", "pred head", "RMSNorm", "mHC gates"],
        "#fff8ee",
        "#E67E22",
    )
    s += _node(
        690,
        cy,
        360,
        84,
        "Muon",
        ["attention / FFN / MoE matrices", "all other W in R(n x m)"],
        "#eef4fc",
        "#4A90D9",
    )
    s += txt(MID, y0 + ph_h - 12, "reuse AdamW LR schedule via RMS rescale gamma = 0.18", "an")

    # --- middle: one Muon step ---
    y1, ph2_h = y0 + ph_h + 14, 168
    s += phase(24, y1, 872, ph2_h, "Algorithm 1 — one Muon step per weight matrix W", "#27AE60")
    row_y = y1 + 78
    nodes = [
        (85, "G_t", ["grad L"], "#f8f9fc", "#c5cee0"),
        (210, "M_t", ["mu M + G"], "#f8f9fc", "#c5cee0"),
        (335, "Nesterov", ["mu M + G"], "#f5f0ff", "#7c3aed"),
        (460, "Hybrid NS", ["10 iter", "8 fast + 2 stable"], "#eef4fc", "#4A90D9"),
        (585, "RMS rescale", ["sqrt max(n,m)", "x gamma"], "#f0faf0", "#27AE60"),
        (710, "W_t", ["decay + update"], "#fff8ee", "#E67E22"),
    ]
    prev_x = None
    box_w = 100
    for cx, title, lines, fill, stroke in nodes:
        s += _node(cx, row_y, box_w, 58, title, lines, fill, stroke)
        if prev_x is not None:
            s += line(prev_x + box_w // 2, row_y, cx - box_w // 2, row_y, "arr-b")
        prev_x = cx

    s += txt(
        MID,
        y1 + ph2_h - 18,
        "NS target: M = U Sigma V^T  ->  UV^T  (polar factor; singular values -> 1)",
        "an",
    )

    # --- bottom: NS stages ---
    y2, ph3_h = y1 + ph2_h + 14, 108
    s += phase(24, y2, 872, ph3_h, "Hybrid Newton-Schulz (V4 tweak vs Kimi Muon)", "#7c3aed")
    cy2 = y2 + 62
    s += _node(
        250,
        cy2,
        380,
        58,
        "Stage 1 - steps 1-8",
        ["(a,b,c) = (3.4445, -4.7750, 2.0315)", "rapid sigma -> 1"],
        "#eef4fc",
        "#4A90D9",
    )
    s += line(440, cy2, 480, cy2, "arr")
    s += _node(
        670,
        cy2,
        380,
        58,
        "Stage 2 - steps 9-10",
        ["(a,b,c) = (2, -1.5, 0.5)", "stabilize sigma = 1"],
        "#f0faf0",
        "#27AE60",
    )

    styles = f"""
<style><![CDATA[
  .an {{ font: 9px sans-serif; fill: #2563eb; text-anchor: middle; }}
  .arr-b {{ stroke: #4A90D9; stroke-width: 1.5; fill: none; marker-end: url(#{PF}ab); }}
]]></style>
"""
    body = styles + s
    for out_dir in (OUT, FIG_V4):
        write("muon-optimizer.svg", body, W, H, PF, out_dir=out_dir)
    print(f"Wrote muon-optimizer.svg -> {FIG_V4}")


if __name__ == "__main__":
    gen_muon_optimizer()
