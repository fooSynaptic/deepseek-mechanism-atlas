#!/usr/bin/env python3
"""DSpark confidence-scheduled verify flow (§6.2)."""
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
SCHED_FILL, SCHED_STROKE = "#fff8ee", "#E67E22"


def gen_dspark_confidence_scheduler() -> None:
    w, h = 920, 588
    pf = "dcs"
    s = ""
    s += txt(460, 22, "DSpark 置信度调度验证", "t")
    s += txt(
        460,
        40,
        "每位置信度 + 硬件吞吐曲线  |  动态选验证长度  |  最大化全局吞吐",
        "st",
    )

    p1_y, p1_h = 56, 124
    s += phase(24, p1_y, 872, p1_h, "步骤 1-2：置信度输出与温度缩放校准")

    cy1 = p1_y + 70
    s += node_auto(150, cy1, 148, "Draft K 位候选", ["semi-AR draft block"], DRAFT_FILL, DRAFT_STROKE)[0]
    s += line(224, cy1, 278, cy1, "arr-g")
    s += node_auto(360, cy1, 168, "逐位置信度", [
        "前缀全接受下",
        "该 token 存活概率",
    ], DRAFT_FILL, DRAFT_STROKE)[0]
    s += line(444, cy1, 490, cy1, "arr-g")
    s += node_auto(580, cy1, 180, "温度缩放校准", [
        "验证集拟合",
        "置信度 约等于 经验接受率",
    ], DRAFT_FILL, DRAFT_STROKE)[0]
    s += txt(460, p1_y + p1_h - 14, "校准后置信度可预测各前缀位被 target 接受的概率", "an")

    p2_y = p1_y + p1_h + 14
    p2_h = 158
    s += phase(24, p2_y, 872, p2_h, "步骤 3：硬件感知前缀调度器")

    in_y = p2_y + 58
    s += node_auto(210, in_y, 168, "并发请求置信度", ["batch 内各请求", "前缀置信曲线"], DRAFT_FILL, DRAFT_STROKE)[0]
    s += node_auto(710, in_y, 168, "实测吞吐曲线", ["GPU batch vs", "verify 长度曲线"], TARGET_FILL, TARGET_STROKE)[0]

    sched_y = p2_y + 118
    sched_top = sched_y - 32
    s += line(210, in_y + 28, 210, sched_top, "arr-g")
    s += line(210, sched_top, 336, sched_top, "arr-g")
    s += line(710, in_y + 28, 710, sched_top, "arr-b")
    s += line(710, sched_top, 584, sched_top, "arr-b")
    s += node_auto(460, sched_y, 248, "硬件感知前缀调度器", [
        "按置信 + 吞吐选验证长度",
        "目标：最大化全局吞吐",
    ], SCHED_FILL, SCHED_STROKE)[0]
    s += txt(
        460,
        p2_y + p2_h - 14,
        "负载自适应：低并发验 4-6 token；高并发平滑缩短，避免 target 争用",
        "an",
    )

    p3_y = p2_y + p2_h + 14
    p3_h = 108
    s += phase(24, p3_y, 872, p3_h, "输出：每请求动态验证长度")

    cy3 = p3_y + 62
    s += line(460, p2_y + p2_h, 460, cy3 - 34, "arr")
    s += node_auto(290, cy3, 156, "低并发", ["verify 4-6 token"], DRAFT_FILL, DRAFT_STROKE)[0]
    s += node_auto(630, cy3, 156, "高并发", ["平滑缩短 verify"], SCHED_FILL, SCHED_STROKE)[0]

    vcy = p3_y + p3_h + 56
    s += line(460, p3_y + p3_h, 460, vcy - 36, "arr-b")
    s += node_auto(460, vcy, 480, "Target verify（选定长度）", [
        "跳过低置信尾巴，少浪费 target 算力",
        "同等吞吐下提升单用户 decode 速度",
    ], TARGET_FILL, TARGET_STROKE)[0]

    for out_dir in (OUT, FIG_RL, FIG_VERSIONS):
        write("dspark-confidence-scheduler.svg", s, w, h, pf, out_dir=out_dir)


if __name__ == "__main__":
    gen_dspark_confidence_scheduler()
