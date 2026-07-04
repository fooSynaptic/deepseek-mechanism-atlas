#!/usr/bin/env python3
"""DeepSeek-V3 FP8 dynamic quantization (paper Figure 7): fine-grained scale + FP32 promotion."""
from __future__ import annotations

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DIAGRAMS, FIGURES, REPO  # noqa: E402

import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
from gen_deepseek_svgs import (  # noqa: E402
    box,
    line,
    node_auto,
    phase,
    txt,
    write,
)

OUT = str(DIAGRAMS)
FIG_V3 = str(FIGURES / "v3")

P1_Y, P1_H = 56, 252
P2_Y, P2_H = P1_Y + P1_H + 18, 240
TOTAL_H = P2_Y + P2_H + 20

TC_FILL, TC_STROKE = "#eef4fc", "#4A90D9"
CUDA_FILL, CUDA_STROKE = "#f0faf0", "#27AE60"
SCALE_FILL, SCALE_STROKE = "#fff8ee", "#E67E22"
DATA_FILL, DATA_STROKE = "#fff7ed", "#fdba74"


def _block_row(s: str, x: int, y: int, n: int, row_label: str) -> tuple[str, int, int]:
    bw, bh, gap = 28, 22, 4
    width = n * bw + (n - 1) * gap
    cx = x + width // 2
    for i in range(n):
        s += box(x + i * (bw + gap), y, bw, bh, DATA_FILL, DATA_STROKE, rx=3)
    s += txt(cx, y - 8, row_label, "an")
    return s, y, y + bh


def gen_v3_fp8_quant() -> None:
    w, h = 920, TOTAL_H
    pf = "fp8"
    s = ""
    s += txt(460, 22, "DeepSeek-V3 FP8 dynamic quantization (Figure 7)", "t")
    s += txt(460, 40, "fine-grained block scales  +  periodic FP32 accumulation promotion", "st")

    # (a) fine-grained quantization
    s += phase(24, P1_Y, 872, P1_H, "(a) Fine-grained quantization  (per-block dynamic scale)")

    ax = 72
    ay = P1_Y + 72
    s, _, abot = _block_row(s, ax, ay, 5, "activation blocks (Nc)")
    sx_y = abot + 14
    for i in range(5):
        s += box(ax + i * 32 + 8, sx_y, 12, 12, SCALE_FILL, SCALE_STROKE, rx=2)
    s += txt(ax + 70, sx_y + 22, "dynamic scale sx", "dt")

    wx = 300
    s, _, wbot = _block_row(s, wx, ay, 5, "weight blocks (Nc)")
    sw_y = wbot + 14
    for i in range(5):
        s += box(wx + i * 32 + 8, sw_y, 12, 12, SCALE_FILL, SCALE_STROKE, rx=2)
    s += txt(wx + 70, sw_y + 22, "dynamic scale sw", "dt")

    tc_x, tc_y = 610, P1_Y + 118
    s += node_auto(tc_x, tc_y, 136, "Tensor Core", ["FP8 GEMM", "low prec product"], TC_FILL, TC_STROKE)[0]
    s += line(ax + 70, abot, tc_x - 68, tc_y, "arr-b")
    s += line(wx + 70, wbot, tc_x - 68, tc_y, "arr-b")

    cu_x = 780
    s += node_auto(cu_x, tc_y, 124, "CUDA Core", ["output", "times sx times sw"], CUDA_FILL, CUDA_STROKE)[0]
    s += line(tc_x + 68, tc_y, cu_x - 62, tc_y, "arr-g")

    s += txt(460, P1_Y + P1_H - 12, "block-wise scales absorb activation outliers before dequant", "an")

    # (b) accumulation precision
    s += phase(24, P2_Y, 872, P2_H, "(b) Increasing accumulation precision  (promote every Nc = 128 MMA)")

    cy = P2_Y + 96
    wgmma_xs = [130, 290, 450, 610]
    for i, x in enumerate(wgmma_xs, 1):
        s += node_auto(x, cy, 88, f"WGMMA {i}", ["FP8 MMA"], TC_FILL, TC_STROKE)[0]
        if i < 4:
            s += line(x + 44, cy, wgmma_xs[i] - 44, cy)

    acc_y = cy + 62
    s += node_auto(370, acc_y, 180, "Tensor Core low prec acc", ["partial sum in TC"], TC_FILL, TC_STROKE)[0]
    s += line(370, cy + 28, 370, acc_y - 28, "arr-b")

    fp_y = acc_y + 62
    s += node_auto(720, fp_y, 156, "CUDA Core FP32 reg", ["promote every Nc", "high prec sum"], CUDA_FILL, CUDA_STROKE)[0]
    s += line(460, acc_y + 28, 642, fp_y - 28, "arr-g")

    s += txt(460, P2_Y + P2_H - 12, "keep FP8 GEMM speed while avoiding drift from long low-prec accumulation", "an")

    for out_dir in (OUT, FIG_V3):
        write("v3-fp8-dynamic-quant.svg", s, w, h, pf, out_dir=out_dir)


if __name__ == "__main__":
    gen_v3_fp8_quant()
