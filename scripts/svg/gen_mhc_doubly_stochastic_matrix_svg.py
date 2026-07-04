#!/usr/bin/env python3
"""Doubly stochastic matrix H: grid + row/col sum = 1 (mHC §3.1)."""
from __future__ import annotations

from pathlib import Path

import sys

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DIAGRAMS, FIGURES, REPO  # noqa: E402

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
sys.path.insert(0, str(ROOT))

from gen_deepseek_svgs import box, line, txt  # noqa: E402
from svg_math_helpers import (  # noqa: E402
    default_math_styles,
    math_line,
    plain,
    t_var,
)

OUT = str(DIAGRAMS)
FIG_MHC = str(FIGURES / "mhc")
PF = "dsm"

W, H = 920, 300

# n=3 numeric example (all rows/cols sum to 1)
N = 3
MATRIX = [
    [0.40, 0.30, 0.30],
    [0.20, 0.50, 0.30],
    [0.40, 0.20, 0.40],
]

CELL_W, CELL_H, GAP = 76, 46, 5
MX, MY = 268, 98


def _styles() -> str:
    return default_math_styles() + """
  .ph { font: 600 12px sans-serif; fill: #4a5a7a; }
  .cell { font: 600 13px Cambria Math, STIX Two Math, serif; fill: #1a1a2e; text-anchor: middle; }
  .idx { font: 10px sans-serif; fill: #666; text-anchor: middle; }
  .sum { font: 600 12px Cambria Math, STIX Two Math, serif; fill: #1d6b3a; text-anchor: middle; }
  .arr { stroke: #888; stroke-width: 1.2; fill: none; marker-end: url(#dsma); }
  .br-r { stroke: #4A90D9; stroke-width: 1.4; fill: none; }
  .br-c { stroke: #E67E22; stroke-width: 1.4; fill: none; }
"""


def _header(w: int, h: int) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
<style><![CDATA[
{_styles()}
]]></style>
<defs>
  <marker id="{PF}a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#888"/></marker>
</defs>
'''


def _matrix_grid() -> str:
    s: list[str] = []
    grid_w = N * CELL_W + (N - 1) * GAP
    grid_h = N * CELL_H + (N - 1) * GAP

    s.append(box(MX - 8, MY - 28, grid_w + 16, grid_h + 16, fill="#fafbfd", stroke="#c5cee0"))

    # column indices j
    for j in range(N):
        cx = MX + j * (CELL_W + GAP) + CELL_W // 2
        s.append(txt(cx, MY - 10, f"j={j + 1}", cls="idx"))

    for i in range(N):
        cy0 = MY + i * (CELL_H + GAP)
        s.append(txt(MX - 22, cy0 + CELL_H // 2 + 4, f"i={i + 1}", cls="idx"))
        for j in range(N):
            cx = MX + j * (CELL_W + GAP)
            cy = cy0
            val = MATRIX[i][j]
            fill = "#eef4fc" if (i + j) % 2 == 0 else "#f8f9fc"
            s.append(box(cx, cy, CELL_W, CELL_H, fill=fill, stroke="#b8c4dc"))
            s.append(
                f'<text x="{cx + CELL_W // 2}" y="{cy + CELL_H // 2 + 5}" class="cell">'
                f"{val:.2f}</text>"
            )

    # row-sum braces (right)
    rx0 = MX + grid_w + 14
    for i in range(N):
        y0 = MY + i * (CELL_H + GAP)
        yc = y0 + CELL_H // 2
        s.append(
            f'<path class="br-r" d="M{rx0},{y0 + 4} Q{rx0 + 10},{yc} {rx0},{y0 + CELL_H - 4}"/>'
        )
        s.append(txt(rx0 + 28, yc + 5, "= 1", cls="sum"))

    s.append(plain(rx0 + 28, MY - 8, "row sum", cls="an", anchor="middle"))

    # col-sum braces (bottom)
    by0 = MY + grid_h + 12
    for j in range(N):
        x0 = MX + j * (CELL_W + GAP)
        xc = x0 + CELL_W // 2
        s.append(
            f'<path class="br-c" d="M{x0 + 4},{by0} Q{xc},{by0 + 10} {x0 + CELL_W - 4},{by0}"/>'
        )
        s.append(txt(xc, by0 + 28, "= 1", cls="sum"))

    s.append(plain(MX + grid_w // 2, by0 + 44, "col sum", cls="an", anchor="middle"))
    return "\n".join(s)


def _legend() -> str:
    lx, ly = 36, 88
    parts = [
        f'<rect x="{lx}" y="{ly}" width="200" height="148" rx="8" fill="#f6f8fb" stroke="#c5cee0" stroke-width="1.5"/>',
        plain(lx + 12, ly + 22, "Three constraints", cls="ph", anchor="start"),
        math_line(lx + 100, ly + 52, [
            t_var("H", "ij"),
            "<tspan> </tspan><tspan font-family='Cambria Math, serif'>&#x2265;</tspan><tspan> 0</tspan>",
        ]),
        math_line(lx + 100, ly + 82, [
            "<tspan font-family='Cambria Math, serif'>&#x2211;</tspan>",
            t_var("j", ""),
            "<tspan> </tspan>",
            t_var("H", "ij"),
            "<tspan> = 1</tspan>",
        ]),
        math_line(lx + 100, ly + 112, [
            "<tspan font-family='Cambria Math, serif'>&#x2211;</tspan>",
            t_var("i", ""),
            "<tspan> </tspan>",
            t_var("H", "ij"),
            "<tspan> = 1</tspan>",
        ]),
        plain(lx + 12, ly + 132, "each row / col: prob. dist.", cls="dt", anchor="start"),
    ]
    return "".join(parts)


def _right_note() -> str:
    rx = 620
    ry = 88
    return (
        box(rx, ry, 268, 148, fill="#fff8ee", stroke="#E67E22")
        + plain(rx + 134, ry + 22, "mHC mixing matrix", cls="ph", anchor="middle")
        + math_line(rx + 134, ry + 52, [t_var("H", "pre")], anchor="middle")
        + math_line(rx + 134, ry + 78, [t_var("H", "post")], anchor="middle")
        + math_line(rx + 134, ry + 104, [t_var("H", "comb")], anchor="middle")
        + plain(rx + 134, ry + 130, "project to Birkhoff via Sinkhorn", cls="dt", anchor="middle")
    )


def gen() -> None:
    body = [
        plain(W // 2, 24, "Doubly stochastic matrix H", cls="t", anchor="middle"),
        plain(W // 2, 44, "example n=3 (mHC uses n=4); entries non-negative, each row and column sums to 1", cls="st", anchor="middle"),
        _legend(),
        _matrix_grid(),
        _right_note(),
    ]
    content = _header(W, H) + "\n".join(body) + "\n</svg>"

    for out_dir in (OUT, FIG_MHC):
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, "mhc-doubly-stochastic-matrix.svg")
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
        ET.parse(p)
        sys.path.insert(0, OUT)
        from svg_validate import validate_svg  # noqa: E402

        errs = validate_svg(Path(p))
        if errs:
            raise SystemExit(f"FAIL {p}: {', '.join(errs)}")
        print(f"OK {p}")


if __name__ == "__main__":
    gen()
