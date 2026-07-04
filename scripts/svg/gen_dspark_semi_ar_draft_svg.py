#!/usr/bin/env python3
"""DSpark semi-AR draft: parallel backbone + sequential head (§6.1)."""
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
FIG_RL = str(FIGURES / "rl")
FIG_VERSIONS = str(REPO / "docs" / "versions" / "figures")

TARGET_FILL, TARGET_STROKE = "#eef4fc", "#4A90D9"
DRAFT_FILL, DRAFT_STROKE = "#f0faf0", "#27AE60"
TOK_FILL, TOK_STROKE = "#fafafa", "#9aa8bc"


def _token_slots(cx_start: int, cy: int, labels: list[str]) -> str:
    s = ""
    n = len(labels)
    gap = 58
    w = 48
    xs = [cx_start + i * gap for i in range(n)]
    for x, lab in zip(xs, labels):
        s += box(x - w // 2, cy - 22, w, 44, TOK_FILL, TOK_STROKE)
        s += txt(x, cy - 6, lab, "lb")
        s += txt(x, cy + 10, "logit", "dt")
    return s


def gen_dspark_semi_ar_draft() -> None:
    w, h = 920, 508
    pf = "dar"
    s = ""
    s += txt(460, 22, "DSpark 半自回归候选生成（并行主干 + 顺序模块）", "t")
    s += txt(
        460,
        40,
        "单轮 draft：一次并行猜 K 位  |  顺序头逐位补块内因果  |  提议 q 对齐 target 校验 p",
        "st",
    )

    # --- Phase 1: parallel backbone ---
    p1_y, p1_h = 56, 172
    s += phase(24, p1_y, 872, p1_h, "阶段 1：并行主干（单轮 draft 一次前向猜 K 位）")

    cy1 = p1_y + 92
    s += node_auto(118, cy1, 128, "Target hidden", ["h from target", "fixed prefix"], TARGET_FILL, TARGET_STROKE)[0]
    s += line(182, cy1, 248, cy1, "arr-b")
    s += node_auto(348, cy1, 168, "并行 Transformer", ["1 forward", "K slots parallel"], DRAFT_FILL, DRAFT_STROKE)[0]
    s += line(432, cy1, 478, cy1, "arr-g")
    s += _token_slots(548, cy1, ["pos 1", "pos 2", "pos 3", "...", "pos K"])
    s += txt(
        460,
        p1_y + p1_h - 14,
        "各位共享 h，块内互不见已猜 draft（第 1 位接近 DFlash，后缀 logits 待修）",
        "an",
    )

    # --- Phase 2: sequential head ---
    p2_y = p1_y + p1_h + 16
    p2_h = 148
    s += phase(24, p2_y, 872, p2_h, "阶段 2：轻量顺序模块（马尔可夫 / RNN 头，逐位注入前序 draft）")

    cy2 = p2_y + 74
    s += node_auto(130, cy2, 108, "draft x1", ["from pos 1"], DRAFT_FILL, DRAFT_STROKE)[0]
    s += line(184, cy2, 228, cy2, "arr-g")
    s += node_auto(292, cy2, 112, "顺序头", ["inject x1"], DRAFT_FILL, DRAFT_STROKE)[0]
    s += line(348, cy2, 392, cy2, "arr-g")
    s += node_auto(456, cy2, 108, "refine pos 2", ["q2"], DRAFT_FILL, DRAFT_STROKE)[0]
    s += line(510, cy2, 554, cy2, "arr-g")
    s += node_auto(618, cy2, 112, "顺序头", ["inject x2"], DRAFT_FILL, DRAFT_STROKE)[0]
    s += line(674, cy2, 718, cy2, "arr-g")
    s += node_auto(782, cy2, 108, "refine pos 3", ["q3 ... qK"], DRAFT_FILL, DRAFT_STROKE)[0]
    s += txt(
        460,
        p2_y + p2_h - 14,
        "x1 注入 pos2，x2 注入 pos3 ... 后缀补块内因果",
        "an",
    )

    vcy = p2_y + p2_h + 78
    s += line(460, p2_y + p2_h, 460, vcy - 38, "arr-b")
    s += node_auto(460, vcy, 500, "Target 校验（ground truth p）", [
        "逐位比较 draft q_i 与 target p_i",
        "接受最长前缀（lossless rejection sampling）",
        "对齐：h 经 f_spec 得 q，校验整段前向得 p",
    ], TARGET_FILL, TARGET_STROKE)[0]

    for out_dir in (OUT, FIG_RL, FIG_VERSIONS):
        write("dspark-semi-ar-draft.svg", s, w, h, pf, out_dir=out_dir)


if __name__ == "__main__":
    gen_dspark_semi_ar_draft()
