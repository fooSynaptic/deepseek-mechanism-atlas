#!/usr/bin/env python3
"""V3.1-Terminus MLA mode switch: Prefill MHA vs Decode MQA (still MLA latent cache)."""
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

LX, RX = 234, 686
MID = 460

P1_Y, P1_H = 56, 352
P2_Y, P2_H = P1_Y + P1_H + 18, 108
TOTAL_H = P2_Y + P2_H + 20

PREFILL_FILL, PREFILL_STROKE = "#eef4fc", "#4A90D9"
DECODE_FILL, DECODE_STROKE = "#f0faf0", "#27AE60"
LAT_FILL, LAT_STROKE = "#fff8ee", "#E67E22"
CACHE_FILL, CACHE_STROKE = "#f8f9fc", "#c5cee0"


def _mini_heads(s: str, cx: int, y: int) -> tuple[str, int, int]:
    hw, hh, gap = 36, 28, 8
    x0 = cx - (4 * hw + 3 * gap) // 2
    for i in range(1, 5):
        x = x0 + (i - 1) * (hw + gap)
        s += box(x, y, hw, hh, PREFILL_FILL, PREFILL_STROKE, rx=4)
        s += txt(x + hw // 2, y + 18, f"Q{i}", "dt")
    s += txt(cx + 92, y + hh + 12, "x H heads", "an")
    return s, y, y + hh


def _mini_latents_split(s: str, cx: int, y: int) -> tuple[str, int, int]:
    hw, hh, gap = 36, 26, 8
    x0 = cx - (4 * hw + 3 * gap) // 2
    for i in range(1, 5):
        x = x0 + (i - 1) * (hw + gap)
        s += box(x, y, hw, hh, LAT_FILL, LAT_STROKE, rx=4)
        s += txt(x + hw // 2, y + 17, f"L{i}", "dt")
    return s, y, y + hh


def _vlink(s: str, cx: int, y_from: int, y_to: int, cls: str = "arr-b") -> str:
    if y_to > y_from + 2:
        s += line(cx, y_from, cx, y_to, cls)
    return s


def gen_mla_mode_switch() -> None:
    w, h = 920, TOTAL_H
    pf = "mla"
    s = ""
    s += txt(MID, 22, "V3.1-Terminus: MLA mode switch by inference phase", "t")
    s += txt(MID, 40, "same MLA weights and latent cache  |  per-head latent at Prefill, shared at Decode", "st")

    s += phase(24, P1_Y, 420, P1_H, "Prefill phase  (MHA-style)")
    s += phase(476, P1_Y, 420, P1_H, "Decode phase  (MQA-style)")

    frag, _, _, _, p_bot = node_auto(
        LX, P1_Y + 96, 188, "Prompt batch", ["L tokens parallel"], PREFILL_FILL, PREFILL_STROKE,
    )
    s += frag

    hq_top = p_bot + 20
    s, _, hq_bot = _mini_heads(s, LX, hq_top)
    s += txt(LX, hq_top - 8, "H query heads", "an")
    s = _vlink(s, LX, p_bot, hq_top)

    s += txt(LX, hq_bot + 16, "each Qi maps to own Li", "lb")

    lat_top = hq_bot + 34
    s, _, lat_bot = _mini_latents_split(s, LX, lat_top)
    s += txt(LX, lat_bot + 12, "H independent latent rows", "an")

    frag, _, c_top, _, c_bot = node_auto(
        LX, lat_bot + 52, 196, "Latent KV write",
        ["MHA-style expansion", "compute-bound prefill"],
        CACHE_FILL, CACHE_STROKE,
    )
    s += frag
    s = _vlink(s, LX, lat_bot + 12, c_top)
    s += txt(LX, c_bot + 12, "bottleneck: compute", "an")

    frag, _, _, _, d_bot = node_auto(
        RX, P1_Y + 96, 188, "Single token", ["one decode step"], DECODE_FILL, DECODE_STROKE,
    )
    s += frag

    dq_top = d_bot + 20
    s, _, dq_bot = _mini_heads(s, RX, dq_top)
    s = _vlink(s, RX, d_bot, dq_top, "arr-g")

    s += txt(RX, dq_bot + 16, "all Qi share one L", "lb")

    frag, _, _, _, sh_bot = node_auto(
        RX, dq_bot + 50, 132, "Shared latent L", ["MQA-style 1 row"], LAT_FILL, LAT_STROKE,
    )
    s += frag

    frag, _, rd_top, _, rd_bot = node_auto(
        RX, sh_bot + 48, 196, "Latent KV read",
        ["full-length cache read", "bandwidth-bound decode"],
        CACHE_FILL, CACHE_STROKE,
    )
    s += frag
    s = _vlink(s, RX, sh_bot, rd_top, "arr-g")
    s += txt(RX, rd_bot + 12, "bottleneck: KV bandwidth", "an")

    s += txt(MID, P1_Y + 210, "engine switches by phase", "st")

    s += phase(24, P2_Y, 872, P2_H, "Still MLA (not standard GQA KV)  |  paves DSA on shared decode latent")
    fy = P2_Y + 62
    for cx, title, lines in [
        (180, "Prefill", ["per-head latent", "like MHA 8 to 8"]),
        (460, "Decode", ["shared latent", "like MQA 8 to 1"]),
        (740, "Not middle GQA", ["no 8-to-2 grouping", "Terminus engine rule"]),
    ]:
        s += node_auto(cx, fy, 248, title, lines, "#f8f9fc", "#c5cee0")[0]

    for out_dir in (OUT, FIG_V3):
        write("mla-mode-switch.svg", s, w, h, pf, out_dir=out_dir)


if __name__ == "__main__":
    gen_mla_mode_switch()
