#!/usr/bin/env python3
"""DSpark speculative decoding: semi-AR draft + confidence-scheduled verify (V4 production)."""
from __future__ import annotations

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DIAGRAMS, FIGURES, REPO  # noqa: E402

import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
from gen_deepseek_svgs import (  # noqa: E402
    line,
    node_auto,
    phase,
    txt,
    write,
)

OUT = str(DIAGRAMS)
FIG_RL = str(FIGURES / "rl")
FIG_VERSIONS = str(REPO / "docs" / "versions" / "figures")

TARGET_FILL, TARGET_STROKE = "#eef4fc", "#4A90D9"
DRAFT_FILL, DRAFT_STROKE = "#f0faf0", "#27AE60"
OTHER_FILL, OTHER_STROKE = "#fff8ee", "#E67E22"


def gen_dspark_speculative() -> None:
    w, h = 920, 420
    pf = "dsp"
    s = ""
    s += txt(460, 22, "DSpark speculative decoding (DeepSeek-V4 production)", "t")
    s += txt(460, 40, "semi-autoregressive draft + confidence-scheduled verify  |  not a new base model", "st")

    p1_y, p1_h = 56, 200
    s += phase(24, p1_y, 872, p1_h, "目标权重上的 DSpark 循环（V4 Flash / Pro + DSpark 模块）")

    cy = p1_y + 108
    s += node_auto(170, cy, 200, "Semi-AR draft", [
        "parallel candidate tokens",
        "higher accept length vs 1-step",
    ], DRAFT_FILL, DRAFT_STROKE)[0]
    s += line(270, cy, 350, cy, "arr-g")
    s += node_auto(460, cy, 220, "Confidence scheduler", [
        "load-aware verify batch",
        "skip low-confidence stalls",
    ], DRAFT_FILL, DRAFT_STROKE)[0]
    s += line(570, cy, 650, cy, "arr-b")
    s += node_auto(750, cy, 200, "Target verify", [
        "same V4 weights",
        "accept longest prefix",
    ], TARGET_FILL, TARGET_STROKE)[0]
    s += txt(460, p1_y + p1_h - 14, "claimed up to ~80% decode speedup vs single-token baseline in production", "an")

    p2_y = p1_y + p1_h + 18
    p2_h = 132
    s += phase(24, p2_y, 872, p2_h, "DSpark 与其他草稿路径的关系")

    row_y = p2_y + 72
    s += node_auto(180, row_y, 200, "V3 MTP native", [
        "MTP heads on same ckpt",
        "see mtp-speculative.svg",
    ], TARGET_FILL, TARGET_STROKE)[0]
    s += node_auto(460, row_y, 200, "DSpark (V4)", [
        "DeepSpec train/eval",
        "online on V4 traffic",
    ], DRAFT_FILL, DRAFT_STROKE)[0]
    s += node_auto(740, row_y, 200, "External draft", [
        "Eagle3 / DFlash / small LM",
        "see spec-decode report",
    ], OTHER_FILL, OTHER_STROKE)[0]
    s += txt(460, p2_y + p2_h - 12, "DeepSpec repo: DSpark + DFlash + Eagle3 draft trainers", "an")

    for out_dir in (OUT, FIG_RL, FIG_VERSIONS):
        write("dspark-speculative.svg", s, w, h, pf, out_dir=out_dir)


if __name__ == "__main__":
    gen_dspark_speculative()
