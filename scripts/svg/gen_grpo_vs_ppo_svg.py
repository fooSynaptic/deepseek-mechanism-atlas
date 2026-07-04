#!/usr/bin/env python3
"""PPO (RLHF) vs GRPO (RLVR) — side-by-side optimization flow comparison."""
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

LX, RX = 234, 686
MID = 460

PANEL_Y, PANEL_H = 56, 448
FOOTER_Y, FOOTER_H = PANEL_Y + PANEL_H + 18, 96
TOTAL_H = FOOTER_Y + FOOTER_H + 20

# PPO warm / GRPO cool palette (match v3-moe-vs-v2)
PPO_FILL, PPO_STROKE = "#fff7ed", "#fdba74"
GRPO_FILL, GRPO_STROKE = "#eef4fc", "#4A90D9"
WARN_FILL, WARN_STROKE = "#fef2f2", "#f87171"
OK_FILL, OK_STROKE = "#f0faf0", "#27AE60"


def _flow_column(
    s: str,
    cx: int,
    steps: list[tuple[str, list[str], str, str]],
    ys: list[int],
) -> str:
    prev_bottom = None
    for cy, (name, lines, fill, stroke) in zip(ys, steps):
        frag, _, y0, _, y1 = node_auto(cx, cy, 196, name, lines, fill, stroke)
        s += frag
        if prev_bottom is not None:
            s += line(cx, prev_bottom, cx, y0)
        prev_bottom = y1
    return s


def _grpo_rollouts(s: str, cx: int, cy: int) -> str:
    w, h = 196, 72
    x, y = cx - w // 2, cy - h // 2
    s += box(x, y, w, h, GRPO_FILL, GRPO_STROKE)
    s += txt(cx, y + 16, "G rollouts / prompt", "lb")
    s += txt(cx, y + 30, "R1 stage-1: G about 16", "dt")
    mini_w, mini_h, gap = 34, 22, 6
    total = 4 * mini_w + 3 * gap
    x0 = cx - total // 2
    my = y + 44
    for i, lab in enumerate(("y1", "y2", "y3", "yG")):
        mx = x0 + i * (mini_w + gap)
        s += box(mx, my, mini_w, mini_h, "#ffffff", GRPO_STROKE, rx=4)
        s += txt(mx + mini_w // 2, my + 15, lab, "dt")
    return s, y, y + h


def gen_grpo_vs_ppo() -> None:
    w, h = 920, TOTAL_H
    pf = "gpo"
    s = ""
    s += txt(MID, 22, "PPO vs GRPO: RL optimization comparison", "t")
    s += txt(MID, 40, "RLHF (neural RM + critic)  vs  RLVR (verifier + group baseline, no critic)", "st")

    s += phase(24, PANEL_Y, 420, PANEL_H, "RLHF + PPO (typical open alignment)")
    s += phase(476, PANEL_Y, 420, PANEL_H, "RLVR + GRPO (DeepSeek-R1 path)")

    s += txt(LX, PANEL_Y + 52, "neural RM reward", "an")
    s += txt(RX, PANEL_Y + 52, "verifier reward", "an")
    s += txt(LX, PANEL_Y + 66, "needs critic V", "an")
    s += txt(RX, PANEL_Y + 66, "no critic", "an")

    ppo_ys = [118, 186, 254, 322, 390, 458]
    ppo_steps = [
        ("Prompt", ["x"], PPO_FILL, PPO_STROKE),
        ("Policy pi", ["LLM weights"], PPO_FILL, PPO_STROKE),
        ("Rollout", ["1 response typical", "per update step"], PPO_FILL, PPO_STROKE),
        ("Neural RM", ["human-preference model", "scalar reward R"], WARN_FILL, WARN_STROKE),
        ("Critic V", ["value network", "extra GPU memory"], WARN_FILL, WARN_STROKE),
        ("PPO update", ["Adv = R - V(s) or GAE", "clip + critic loss"], PPO_FILL, PPO_STROKE),
    ]
    s = _flow_column(s, LX, ppo_steps, ppo_ys)

    grpo_ys_pre = [118, 186]
    grpo_pre = [
        ("Prompt", ["x"], GRPO_FILL, GRPO_STROKE),
        ("Policy pi", ["V3-Base weights"], GRPO_FILL, GRPO_STROKE),
    ]
    s = _flow_column(s, RX, grpo_pre, grpo_ys_pre)

    # rollouts custom block
    cy_roll = 254
    s += line(RX, 186 + 28, RX, cy_roll - 36)
    s, _, roll_bottom = _grpo_rollouts(s, RX, cy_roll)

    grpo_ys_rest = [322, 390, 458]
    grpo_rest = [
        ("Verifier reward", ["sympy / unit test / format", "scores r1..rG"], OK_FILL, OK_STROKE),
        ("Group baseline", ["mean r over G samples", "Adv_i = r_i - mean(r)"], OK_FILL, OK_STROKE),
        ("GRPO update", ["policy grad only", "KL to reference + clip"], GRPO_FILL, GRPO_STROKE),
    ]
    prev_bottom = roll_bottom
    for cy, (name, lines, fill, stroke) in zip(grpo_ys_rest, grpo_rest):
        frag, _, box_y0, _, box_y1 = node_auto(RX, cy, 196, name, lines, fill, stroke)
        s += line(RX, prev_bottom, RX, box_y0)
        s += frag
        prev_bottom = box_y1

    # center contrast callouts
    s += txt(MID, 312, "vs", "st")
    s += txt(MID, 380, "V(s) baseline", "an")
    s += txt(MID, 396, "group mean baseline", "an")

    # footer summary
    s += phase(24, FOOTER_Y, 872, FOOTER_H, "Key contrast")
    fy = FOOTER_Y + 58
    cols = [
        (180, "Critic", ["PPO: value net V", "GRPO: none"]),
        (460, "Advantage", ["PPO: R - V or GAE", "GRPO: r_i - mean(r)"]),
        (740, "Typical use", ["PPO: open alignment", "GRPO: math / code RLVR"]),
    ]
    for cx, title, lines in cols:
        s += node_auto(cx, fy, 248, title, lines, "#f8f9fc", "#c5cee0")[0]

    for out_dir in (OUT, FIG_RL):
        write("grpo-vs-ppo.svg", s, w, h, pf, out_dir=out_dir)


if __name__ == "__main__":
    gen_grpo_vs_ppo()
