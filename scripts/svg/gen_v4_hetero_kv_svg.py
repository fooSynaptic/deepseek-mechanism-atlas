#!/usr/bin/env python3
"""V4 heterogeneous KV: CSA 4:1, HCA 128:1, SWA, Indexer, tail + HiSparse offload."""
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
FIG_V4 = str(FIGURES / "v4")

W = 920
MID = W // 2


def gen_v4_hetero_kv() -> None:
    y0 = 56
    comp_h = 108
    cache_h = 148
    his_h = 96
    y2 = y0 + comp_h + 14 + cache_h + 14
    h = y2 + his_h + 20

    s = ""
    s += txt(MID, 22, "DeepSeek-V4: heterogeneous KV cache (CSA / HCA / SWA)", "t")
    s += txt(MID, 40, "1M context: multi-type entries replace single MLA latent stream", "st")

    s += phase(24, y0, 872, comp_h, "Token stream compression (algorithm)", "#E67E22")
    cy = y0 + 58
    s += node_auto(180, cy, 220, "Raw tokens L", ["full sequence"], "#f8f9fc", "#c5cee0")[0]
    s += node_auto(460, cy, 240, "CSA 4:1 compress", ["1 entry / 4 tokens", "then DSA top-k"], "#eef4fc", "#4A90D9")[0]
    s += node_auto(740, cy, 220, "HCA 128:1 compress", ["1 entry / 128 tokens", "short dense attend"], "#f0faf0", "#27AE60")[0]
    s += line(290, cy, 340, cy)
    s += line(580, cy, 630, cy)

    y1 = y0 + comp_h + 14
    s += phase(24, y1, 872, cache_h, "Coexisting KV types at inference", "#4A90D9")
    row_y = y1 + 52
    types = [
        (120, "CSA entry", ["4:1 compressed", "sparse top-k"], "#eef4fc", "#4A90D9"),
        (280, "HCA entry", ["128:1 compressed", "dense short seq"], "#f0faf0", "#27AE60"),
        (440, "SWA window", ["recent n_win", "own eviction"], "#fff8ee", "#E67E22"),
        (600, "Indexer KV", ["CSA lightning", "importance scores"], "#f5f0ff", "#7c3aed"),
        (760, "Tail buffer", ["incomplete block", "wait for m"], "#f8f9fc", "#c5cee0"),
    ]
    for cx, title, lines, fill, stroke in types:
        s += node_auto(cx, row_y, 130, title, lines, fill, stroke)[0]
    s += txt(MID, y1 + cache_h - 14, "Classical KV cache + per-request state blocks (paper sec 3.5.1)", "an")

    y2 = y1 + cache_h + 14
    s += phase(24, y2, 872, his_h, "HiSparse offload (infra): inactive C4 entries to CPU pinned memory", "#27AE60")
    hy = y2 + 50
    s += node_auto(260, hy, 200, "GPU hot set", ["active CSA entries", "decode working set"], "#eef4fc", "#4A90D9")[0]
    s += txt(460, hy + 4, "prefetch", "lb")
    s += node_auto(660, hy, 220, "CPU pinned pool", ["inactive C4 offloaded", "~3x token capacity"], "#f0faf0", "#27AE60")[0]
    s += line(360, hy, 550, hy, "arr-g")

    pf = "v4"
    for out_dir in (OUT, FIG_V4):
        write("v4-hetero-kv.svg", s, W, h, pf, out_dir=out_dir)


if __name__ == "__main__":
    gen_v4_hetero_kv()
