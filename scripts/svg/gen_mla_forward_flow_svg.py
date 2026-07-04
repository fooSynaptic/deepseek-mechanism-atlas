#!/usr/bin/env python3
"""MLA forward flow (Eq. 37-47): compact overview (Q/K/V text columns + cache contrast).

The canonical detailed figure with three bottom-left callout boxes lives at
docs/figures/mla/mla-forward-flow.svg. This generator only maintains mla-forward-flow-compact.svg.
"""
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
FIG_MLA = str(FIGURES / "mla")

W = 920
COL_W = 276
COL_GAP = 34
X0 = 24
X1 = X0 + COL_W + COL_GAP
X2 = X1 + COL_W + COL_GAP
MID = W // 2

Q_FILL, Q_STROKE = "#eef4fc", "#4A90D9"
K_FILL, K_STROKE = "#f0faf0", "#27AE60"
V_FILL, V_STROKE = "#fff8ee", "#E67E22"
ATT_FILL, ATT_STROKE = "#f5f0ff", "#7c3aed"
CACHE_FILL, CACHE_STROKE = "#f8f9fc", "#c5cee0"


def _col_lines(x: int, y: int, w: int, title: str, lines: list[str], fill: str, stroke: str) -> str:
    h = 28 + len(lines) * 15
    s = phase(x, y, w, h, title, stroke)
    for i, ln in enumerate(lines):
        s += txt(x + w // 2, y + 38 + i * 15, ln, "dt")
    return s


def gen_mla_forward_flow() -> None:
    col_h = 152
    y_cols = 96
    y_attn = y_cols + col_h + 22
    attn_h = 68
    y_out = y_attn + attn_h + 28
    out_h = 52
    y_cache = y_out + out_h + 16
    cache_h = 118
    h = y_cache + cache_h + 18

    s = ""
    s += txt(MID, 22, "MLA forward flow (Multi-Head Latent Attention, Eq. 37-47)", "t")
    s += txt(MID, 40, "DeepSeek-V2/V3 typical dims: d=5120, n_h=128, d_c^KV=512, d_h^R=64", "st")

    frag, _, ht_top, _, ht_bot = node_auto(
        MID, 68, 168, "Input h_t", ["[B, d_model]"], "#f8f9fc", "#334155"
    )
    s += frag

    s += _col_lines(
        X0,
        y_cols,
        COL_W,
        "Query branch (37-40)",
        [
            "W^DQ h_t -> c_t^Q [1536]",
            "W^UQ c_t^Q -> q_t^C [128,128]",
            "RoPE(W^QR c_t^Q) -> q_t^R",
            "all heads share q_t^R [128,64]",
            "q_{t,i} = [q_{t,i}^C ; q_t^R]",
        ],
        Q_FILL,
        Q_STROKE,
    )
    s += _col_lines(
        X1,
        y_cols,
        COL_W,
        "Key branch (41-44)",
        [
            "W^DKV h_t -> c_t^KV [512]",
            "K and V share c_t^KV",
            "W^UK c_t^KV -> k_t^C per head",
            "RoPE(W^KR h_t) -> k_t^R [64]",
            "k_{t,i} = [k_{t,i}^C ; k_t^R]",
        ],
        K_FILL,
        K_STROKE,
    )
    s += _col_lines(
        X2,
        y_cols,
        COL_W,
        "Value branch (45)",
        [
            "reuse c_t^KV from Key",
            "W^UV c_t^KV -> v_t^C",
            "no RoPE / no R branch",
            "v_{t,i}^C per head [128]",
            "simplest path saves cache",
        ],
        V_FILL,
        V_STROKE,
    )

    for cx in (X0 + COL_W // 2, X1 + COL_W // 2, X2 + COL_W // 2):
        s += line(MID, ht_bot, cx, y_cols, "arr-b" if cx == X0 + COL_W // 2 else "arr-g" if cx == X1 + COL_W // 2 else "arr")

    s += phase(24, y_attn, 872, attn_h, "Attention per head i (46)", ATT_STROKE)
    ay = y_attn + 44
    s += node_auto(130, ay, 108, "q_{t,i}", ["[d_h]"], Q_FILL, Q_STROKE)[0]
    s += txt(250, ay + 4, "dot k_{j,i}", "lb")
    s += node_auto(360, ay, 108, "k_{j,i}", ["j=1..t"], K_FILL, K_STROKE)[0]
    s += txt(490, ay + 4, "Softmax", "lb")
    s += node_auto(590, ay, 108, "v_{j,i}^C", ["[d_h^C]"], V_FILL, V_STROKE)[0]
    s += line(680, ay, 730, ay)
    s += node_auto(790, ay, 108, "o_{t,i}", ["[d_h^C]"], ATT_FILL, ATT_STROKE)[0]
    s += txt(MID, y_attn + attn_h + 16, "weighted sum uses v^C only (no R in value)", "an")

    s += phase(24, y_out, 872, out_h, "Multi-head output (47)", ATT_STROKE)
    s += txt(MID, y_out + 34, "concat(o_{t,1},...,o_{t,n_h}) then W^O -> u_t [B, d_model]", "lb")

    s += phase(24, y_cache, 872, cache_h, "KV cache at inference (typical dims)", CACHE_STROKE)
    cy = y_cache + 62
    s += box(48, cy - 36, 380, 72, "#fff5f5", "#fca5a5")
    s += txt(238, cy - 18, "Standard MHA cache", "lb")
    s += txt(238, cy - 2, "2 x n_h x d_h^C ~ 32768 dims / token", "dt")
    s += txt(238, cy + 14, "grows linearly with context T", "dt")
    s += txt(460, cy + 4, "vs", "t")
    s += box(492, cy - 36, 380, 72, "#f0faf0", "#27AE60")
    s += txt(682, cy - 18, "MLA cache", "lb")
    s += txt(682, cy - 2, "c^KV [512] + shared k^R [64] = 576 / token", "dt")
    s += txt(682, cy + 14, "on read: W^UK / W^UV upsample to heads", "dt")

    pf = "mla"
    for out_dir in (OUT, FIG_MLA):
        write("mla-forward-flow-compact.svg", s, W, h, pf, out_dir=out_dir)


if __name__ == "__main__":
    gen_mla_forward_flow()
