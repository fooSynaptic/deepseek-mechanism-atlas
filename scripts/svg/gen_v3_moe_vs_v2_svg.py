#!/usr/bin/env python3
"""V3 vs V2 MoE structural innovations — three-panel comparison SVG (tspan math)."""
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

from gen_deepseek_svgs import box, line, phase, write  # noqa: E402
from svg_math_helpers import (  # noqa: E402
    default_math_styles,
    math_line,
    plain,
    t_loss,
    t_var,
)

OUT = str(DIAGRAMS)
FIG_V3 = str(FIGURES / "v3")
PF = "v3m"

# vertical rhythm: title 0-52, panels with >=18px gap
P1_Y, P1_H = 64, 198
P2_Y, P2_H = P1_Y + P1_H + 18, 172
P3_Y, P3_H = P2_Y + P2_H + 18, 168
TOTAL_H = P3_Y + P3_H + 20


def _styles() -> str:
    return default_math_styles() + f"""
  .ph {{ font: 600 12px sans-serif; fill: #4a5a7a; }}
  .arr {{ stroke: #666; stroke-width: 1.5; fill: none; marker-end: url(#{PF}a); }}
  .arr-d {{ stroke: #888; stroke-width: 1.2; fill: none; stroke-dasharray: 5 4; marker-end: url(#{PF}ad); }}
  .arr-b {{ stroke: #4A90D9; stroke-width: 1.5; fill: none; marker-end: url(#{PF}ab); }}
  .arr-g {{ stroke: #27AE60; stroke-width: 1.5; fill: none; marker-end: url(#{PF}ag); }}
"""


def _markers() -> str:
    return f"""
  <marker id="{PF}a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#666"/></marker>
  <marker id="{PF}ad" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#888"/></marker>
  <marker id="{PF}ab" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#4A90D9"/></marker>
  <marker id="{PF}ag" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#27AE60"/></marker>
"""


def _header(w: int, h: int) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
<style><![CDATA[
{_styles()}
]]></style>
<defs>
{_markers()}
</defs>
'''


def _write_math(name: str, body: str, w: int, h: int, out_dir: str) -> None:
    content = _header(w, h) + body + "</svg>"
    p = os.path.join(out_dir, name)
    os.makedirs(out_dir, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    ET.parse(p)
    sys.path.insert(0, OUT)
    from svg_validate import validate_svg  # noqa: E402

    layout_errs = validate_svg(Path(p))
    if layout_errs:
        raise SystemExit(f"FAIL {name}: {', '.join(layout_errs)}")
    print("OK", p)


def _box_label(cx: int, cy: int, w: int, h: int, fill: str, stroke: str, title: str, parts: list[str]) -> str:
    x, y = cx - w // 2, cy - h // 2
    s = box(x, y, w, h, fill, stroke)
    s += plain(cx, y + 18, title, cls="lb", anchor="middle")
    for i, row in enumerate(parts):
        s += math_line(cx, y + 34 + i * 14, row, cls="fm", anchor="middle")
    return s


def _box_math_title(cx: int, cy: int, w: int, h: int, fill: str, stroke: str, title_parts: list[str], lines: list[list[str]]) -> str:
    x, y = cx - w // 2, cy - h // 2
    lh = max(56, 36 + len(lines) * 14)
    y = cy - lh // 2
    s = box(cx - w // 2, y, w, lh, fill, stroke)
    s += math_line(cx, y + 18, title_parts, cls="lb", anchor="middle")
    for i, row in enumerate(lines):
        s += math_line(cx, y + 34 + i * 14, row, cls="fm", anchor="middle")
    return s


def _phase_p1_title(x: int, y: int) -> str:
    s = (
        f'<rect x="{x}" y="{y}" width="872" height="{P1_H}" rx="8" fill="#eef1f8" '
        f'stroke="#b0bdd4" stroke-width="1.5" stroke-dasharray="6 3"/>'
    )
    s += plain(x + 12, y + 18, "1  aux-loss-free routing (V2 softmax + ", cls="ph", anchor=None)
    s += math_line(x + 318, y + 18, [t_loss("aux")], cls="m", anchor="start")
    s += plain(x + 348, y + 18, "  vs  V3 sigmoid + bias)", cls="ph", anchor=None)
    return s


def _router_panel(s: str) -> str:
    y0 = P1_Y
    s += _phase_p1_title(24, y0)

    s += plain(228, y0 + 34, "DeepSeek-V2", cls="lb", anchor="middle")
    s += plain(692, y0 + 34, "DeepSeek-V3", cls="lb", anchor="middle")
    s += plain(460, y0 + 34, "vs", cls="st", anchor="middle")

    cy = y0 + 88
    vx = 228
    s += _box_math_title(vx - 88, cy, 68, 56, "#fff7ed", "#fdba74", ["Token"], [[t_var("h", "t")]])
    s += _box_label(vx, cy, 84, 56, "#fff7ed", "#fdba74", "Router", [["<tspan>softmax</tspan>"]])
    s += _box_label(vx + 92, cy, 84, 56, "#fff7ed", "#fdba74", "Top-6", [["<tspan>160 exp.</tspan>"]])
    s += line(vx - 52, cy, vx - 42, cy)
    s += line(vx + 42, cy, vx + 50, cy)

    aux_cy = y0 + 158
    s += _box_math_title(vx, aux_cy, 128, 56, "#fef2f2", "#f87171", [t_loss("aux"), "<tspan> branch</tspan>"], [["<tspan>load balance loss</tspan>"]])
    s += line(vx + 42, cy + 22, vx, aux_cy - 30, "arr-d")
    s += plain(vx, aux_cy + 38, "hurts main task loss", cls="an", anchor="middle")

    ux = 692
    s += _box_math_title(ux - 98, cy, 68, 56, "#eef4fc", "#4A90D9", ["Token"], [[t_var("h", "t")]])
    s += _box_math_title(
        ux,
        cy,
        104,
        56,
        "#eef4fc",
        "#4A90D9",
        ["Router"],
        [
            ["<tspan>sigmoid </tspan>", t_var("s")],
            ["<tspan>+ bias </tspan>", t_var("b", "i")],
        ],
    )
    s += _box_label(ux + 112, cy, 84, 56, "#eef4fc", "#4A90D9", "Top-8", [["<tspan>256 exp.</tspan>"]])
    s += line(ux - 64, cy, ux - 54, cy)
    s += line(ux + 52, cy, ux + 70, cy)

    note_y = y0 + 148
    s += math_line(
        ux,
        note_y,
        [
            "<tspan>select: </tspan>",
            t_var("s"),
            "<tspan>+</tspan>",
            t_var("b", "i"),
            "<tspan>  |  gate: </tspan>",
            t_var("s"),
            "<tspan> only</tspan>",
        ],
        cls="an",
        anchor="middle",
    )
    s += math_line(
        ux,
        note_y + 14,
        ["<tspan>no </tspan>", t_loss("aux"), "<tspan> in forward graph</tspan>"],
        cls="an",
        anchor="middle",
    )
    return s


def _scale_panel(s: str) -> str:
    y0, ph = P2_Y, P2_H
    s += phase(24, y0, 872, ph, "2  expert scale and sparsity (fine-grained routed pool)")

    row_y = y0 + 92
    cols = [
        (228, "V2", [
            "160 routed / layer",
            "Top-6 per token",
            "2 shared / layer",
            "236B / 21B act (~8.9%)",
        ], "#fff7ed", "#fdba74"),
        (692, "V3", [
            "256 routed / layer",
            "Top-8 per token",
            "1 shared / layer",
            "671B / 37B act (~5.5%)",
        ], "#eef4fc", "#4A90D9"),
    ]
    for cx, title, lines_txt, fill, stroke in cols:
        parts = [[f"<tspan>{ln}</tspan>"] for ln in lines_txt]
        s += _box_math_title(cx, row_y, 196, 56, fill, stroke, [f"<tspan>{title}</tspan>"], parts)

    s += plain(460, row_y - 18, "finer pool, lower act %", cls="an", anchor="middle")
    return s


def _fusion_panel(s: str) -> str:
    y0, ph = P3_Y, P3_H
    s += phase(24, y0, 872, ph, "3  shared + routed fusion (FFN output structure)")

    cy = y0 + 86
    s += _box_math_title(
        118,
        cy,
        96,
        56,
        "#f8f9fc",
        "#c5cee0",
        ["<tspan>Hidden </tspan>", t_var("h", "t")],
        [["<tspan>layer in</tspan>"]],
    )

    s += _box_label(300, cy - 38, 124, 56, "#eef4fc", "#4A90D9", "Router path", [
        ["<tspan>sigmoid+bias top-8</tspan>"],
        ["<tspan>256 routed</tspan>"],
    ])

    s += _box_label(300, cy + 38, 124, 56, "#f0faf0", "#27AE60", "Shared path", [
        ["<tspan>always on</tspan>"],
        ["<tspan>no top-K</tspan>"],
    ])

    s += _box_label(510, cy, 116, 56, "#f8f9fc", "#c5cee0", "Weighted sum", [
        ["<tspan>routed + shared</tspan>"],
    ])

    s += _box_label(692, cy, 100, 56, "#f8f9fc", "#c5cee0", "FFN out", [
        ["<tspan>residual</tspan>"],
    ])

    s += line(166, cy, 238, cy - 38)
    s += line(166, cy, 238, cy + 38)
    s += line(362, cy - 38, 452, cy)
    s += line(362, cy + 38, 452, cy)
    s += line(568, cy, 642, cy)
    return s


def gen_v3_moe_vs_v2() -> None:
    w, h = 920, TOTAL_H
    s = ""
    s += plain(460, 22, "DeepSeek-V3 vs V2: MoE structural innovations", cls="t", anchor="middle")
    s += plain(460, 40, "model internals only (routing, expert pool, FFN fusion)", cls="st", anchor="middle")

    s = _router_panel(s)
    s = _scale_panel(s)
    s = _fusion_panel(s)

    for out_dir in (OUT, FIG_V3):
        _write_math("v3-moe-vs-v2.svg", s, w, h, out_dir=out_dir)


if __name__ == "__main__":
    gen_v3_moe_vs_v2()
