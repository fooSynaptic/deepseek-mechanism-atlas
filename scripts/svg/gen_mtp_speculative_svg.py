#!/usr/bin/env python3
"""DeepSeek-V3 MTP architecture + MTP native vs external draft speculative decoding."""
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
FIG_V3 = str(FIGURES / "v3")
FIG_VERSIONS = str(REPO / "docs" / "versions" / "figures")

MAIN_FILL, MAIN_STROKE = "#eef4fc", "#4A90D9"
MTP_FILL, MTP_STROKE = "#f0faf0", "#27AE60"
SHARED_FILL, SHARED_STROKE = "#fff8ee", "#E67E22"
DRAFT_FILL, DRAFT_STROKE = "#fff7ed", "#fdba74"

P1_Y, P1_H = 56, 400
P2_Y, P2_H = P1_Y + P1_H + 18, 176
TOTAL_H = P2_Y + P2_H + 20

STEP = 68  # node_auto min height 56 + gap


def _stack(
    s: str,
    cx: int,
    col_title: str,
    nodes: list[tuple[str, list[str], str, str]],
    y0: int,
) -> str:
    s += txt(cx, P1_Y + 32, col_title, "lb")
    prev_bottom = None
    for i, (name, lines, fill, stroke) in enumerate(nodes):
        cy = y0 + i * STEP
        frag, _, box_y0, _, y1 = node_auto(cx, cy, 168, name, lines, fill, stroke)
        s += frag
        if prev_bottom is not None:
            s += line(cx, prev_bottom, cx, box_y0)
        prev_bottom = y1
    return s


def gen_mtp_speculative() -> None:
    w, h = 920, TOTAL_H
    pf = "mtp"
    s = ""
    s += txt(460, 22, "DeepSeek-V3 MTP and speculative decoding", "t")
    s += txt(460, 40, "training: shared emb/head + causal MTP chain  |  inference: native MTP vs external draft", "st")

    s += phase(24, P1_Y, 872, P1_H, "1  MTP training (V3 paper Figure 3 style)")
    s += txt(460, P1_Y + 52, "hidden chain: Main Block L  then  MTP-1 Block  then  MTP-2 Block", "an")

    main_nodes = [
        ("Input", ["t1 t2 t3 t4"], MAIN_FILL, MAIN_STROKE),
        ("Emb Layer", ["shared"], SHARED_FILL, SHARED_STROKE),
        ("Transformer", ["Block x L"], MAIN_FILL, MAIN_STROKE),
        ("Out Head", ["shared"], SHARED_FILL, SHARED_STROKE),
        ("Target / Loss", ["t2..t5  L_Main"], MAIN_FILL, MAIN_STROKE),
    ]
    mtp_nodes = [
        ("Input", ["shifted tokens"], MTP_FILL, MTP_STROKE),
        ("Emb Layer", ["shared"], SHARED_FILL, SHARED_STROKE),
        ("Integration", ["RMSNorm concat proj"], MTP_FILL, MTP_STROKE),
        ("Transformer", ["1 Block"], MTP_FILL, MTP_STROKE),
        ("Out Head", ["shared"], SHARED_FILL, SHARED_STROKE),
        ("Target / Loss", ["L_MTP-k"], MTP_FILL, MTP_STROKE),
    ]

    y_main = 132
    y_mtp = 122
    s = _stack(s, 160, "Main Model (next token)", main_nodes, y_main)
    s = _stack(s, 460, "MTP Module 1 (next2)", mtp_nodes, y_mtp)
    s = _stack(s, 760, "MTP Module 2 (next3)", mtp_nodes, y_mtp)

    s += txt(460, P1_Y + P1_H - 14, "joint loss: L_Main + sum L_MTP-k", "an")

    s += phase(24, P2_Y, 872, P2_H, "2  Inference: same speculative loop, different draft source")

    s += txt(460, P2_Y + 36, "both: draft proposes k tokens, target verifies, accept prefix", "lb")

    inf_y = P2_Y + 108
    s += node_auto(234, inf_y, 220, "MTP native (V3)", [
        "MTP heads on same ckpt",
        "no extra model load",
    ], MAIN_FILL, MAIN_STROKE)[0]

    s += node_auto(686, inf_y, 220, "External draft (this report)", [
        "Qwen3-0.6B draft",
        "Qwen3-4B target verify",
    ], DRAFT_FILL, DRAFT_STROKE)[0]

    s += txt(460, inf_y + 48, "this benchmark: 2.4-2.9x vs 4B nospec", "an")

    for out_dir in (OUT, FIG_V3, FIG_VERSIONS):
        write("mtp-speculative.svg", s, w, h, pf, out_dir=out_dir)


if __name__ == "__main__":
    gen_mtp_speculative()
