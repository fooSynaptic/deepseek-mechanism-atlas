#!/usr/bin/env python3
"""Hash MoE: gate affinity (V3) vs hash query (V4 shallow) — routing swap diagram."""
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

from gen_deepseek_svgs import box, line, write  # noqa: E402
from svg_math_helpers import (  # noqa: E402
    default_math_styles,
    math_line,
    plain,
    t_arrow,
    t_dot,
    t_greek,
    t_var,
)

OUT = str(DIAGRAMS)
FIG_V4 = str(FIGURES / "v4")
PF = "hmr"

W, H = 920, 460
P1_Y, P1_H = 58, 210
P2_Y, P2_H = P1_Y + P1_H + 16, 132
MID_X = W // 2

_TITLE_Y = 18
_ROW0_Y = 34
_ROW_DY = 13
_PAD_BOTTOM = 10


def _box(
    cx: int,
    cy: int,
    w: int,
    h_min: int,
    fill: str,
    stroke: str,
    title: str,
    rows: list[list[str]],
) -> str:
    """Box with title + math rows; height grows to fit content."""
    n = max(1, len(rows))
    lh = max(h_min, _ROW0_Y + n * _ROW_DY + _PAD_BOTTOM)
    x, y = cx - w // 2, cy - lh // 2
    s = box(x, y, w, lh, fill, stroke)
    s += plain(cx, y + _TITLE_Y, title, cls="lb", anchor="middle")
    for i, parts in enumerate(rows):
        cls = "dt" if n > 1 and i > 0 else "fm"
        s += math_line(cx, y + _ROW0_Y + i * _ROW_DY, parts, cls=cls, anchor="middle")
    return s


def _styles() -> str:
    return default_math_styles() + f"""
  .ph {{ font: 600 12px sans-serif; fill: #4a5a7a; }}
  .arr {{ stroke: #666; stroke-width: 1.5; fill: none; marker-end: url(#{PF}a); }}
  .arr-d {{ stroke: #888; stroke-width: 1.2; fill: none; stroke-dasharray: 5 4; marker-end: url(#{PF}ad); }}
  .arr-b {{ stroke: #4A90D9; stroke-width: 1.5; fill: none; marker-end: url(#{PF}ab); }}
  .arr-o {{ stroke: #E67E22; stroke-width: 1.5; fill: none; marker-end: url(#{PF}ao); }}
  .arr-g {{ stroke: #27AE60; stroke-width: 1.5; fill: none; marker-end: url(#{PF}ag); }}
"""


def _markers() -> str:
    return f"""
  <marker id="{PF}a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#666"/></marker>
  <marker id="{PF}ad" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#888"/></marker>
  <marker id="{PF}ab" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#4A90D9"/></marker>
  <marker id="{PF}ao" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#E67E22"/></marker>
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


def _write_svg(name: str, body: str, w: int, h: int, out_dir: str) -> None:
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


def _phase_frame(x: int, y: int, w: int, h: int, label: str) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#eef1f8" '
        f'stroke="#b0bdd4" stroke-width="1.5" stroke-dasharray="6 3"/>'
        f'<text x="{x + 12}" y="{y + 18}" class="ph">{label}</text>'
    )


def _panel_v3(s: str) -> str:
    x0, y0, pw, ph = 24, P1_Y, 420, P1_H
    s += _phase_frame(x0, y0, pw, ph, "A  V3 gate affinity (centroid router GEMM)")
    s += math_line(
        x0 + pw // 2,
        y0 + 38,
        [
            t_var("s", "i,t"),
            "<tspan> = sigmoid(</tspan>",
            t_var("u", "t"),
            "<tspan> </tspan>",
            t_dot(),
            "<tspan> </tspan>",
            t_var("e", "i"),
            "<tspan>)</tspan>",
        ],
        cls="m",
        anchor="middle",
    )

    cy = y0 + 96
    bx = x0 + 36
    gap = 16
    w1, w2, w3, w4 = 76, 98, 86, 68
    x2 = bx + w1 + gap
    x3 = x2 + w2 + gap
    x4 = x3 + w3 + gap
    s += _box(bx + w1 // 2, cy, w1, 58, "#eef4fc", "#4A90D9", "Token", [[t_var("u", "t")]])
    s += _box(x2 + w2 // 2, cy, w2, 58, "#eef4fc", "#4A90D9", "Affinity", [
        ["<tspan>256 scores</tspan>"],
        [t_var("u", "t"), t_dot(), t_var("e", "i")],
    ])
    s += _box(x3 + w3 // 2, cy, w3, 58, "#eef4fc", "#4A90D9", "Select", [
        ["<tspan>+ </tspan>", t_var("b", "i")],
        ["<tspan>top-</tspan>", t_var("K", "r")],
    ])
    s += _box(x4 + w4 // 2, cy, w4, 58, "#eef4fc", "#4A90D9", "Expert id", [
        ["<tspan>top-</tspan>", t_var("K", "r"), "<tspan> ids</tspan>"],
    ])

    s += line(bx + w1, cy, x2, cy, "arr-b")
    s += line(x2 + w2, cy, x3, cy, "arr-b")
    s += line(x3 + w3, cy, x4, cy, "arr-b")

    ly = y0 + 168
    s += _box(x0 + pw // 2, ly, 340, 52, "#fef2f2", "#f87171", "Learned router params", [
        [t_var("e", "i"), "<tspan> centroid vectors, dynamic </tspan>", t_var("b", "i")],
    ])
    s += plain(x0 + pw // 2, ly + 38, "semantic match: token picks expert prototype", cls="an", anchor="middle")
    return s


def _panel_v4(s: str) -> str:
    x0, y0, pw, ph = 476, P1_Y, 420, P1_H
    s += _phase_frame(x0, y0, pw, ph, "B  V4 Hash query (deterministic, no GEMM)")
    s += math_line(
        x0 + pw // 2,
        y0 + 38,
        [
            t_var("i", "t"),
            "<tspan> = </tspan>",
            t_greek("phi"),
            "<tspan>(</tspan>",
            t_var("x", "t"),
            "<tspan>, </tspan>",
            t_var("t"),
            "<tspan>) mod </tspan>",
            t_var("N", "r"),
        ],
        cls="m",
        anchor="middle",
    )

    cy = y0 + 96
    bx = x0 + 36
    gap = 16
    w1, w2, w3, w4 = 78, 96, 84, 74
    x2 = bx + w1 + gap
    x3 = x2 + w2 + gap
    x4 = x3 + w3 + gap
    s += _box(bx + w1 // 2, cy, w1, 58, "#fff8ee", "#E67E22", "Token", [
        [t_var("x", "t")],
        ["<tspan>pos </tspan>", t_var("t")],
    ])
    s += _box(x2 + w2 // 2, cy, w2, 58, "#fff8ee", "#E67E22", "Hash fn", [
        ["<tspan>no </tspan>", t_var("e", "i")],
        [t_greek("phi"), "<tspan> lookup</tspan>"],
    ])
    s += _box(x3 + w3 // 2, cy, w3, 58, "#fff8ee", "#E67E22", "Mod pool", [
        ["<tspan>mod </tspan>", t_var("N", "r")],
        ["<tspan>static spread</tspan>"],
    ])
    s += _box(x4 + w4 // 2, cy, w4, 58, "#fff8ee", "#E67E22", "Expert id", [
        [t_var("i", "t")],
    ])

    s += line(bx + w1, cy, x2, cy, "arr-o")
    s += line(x2 + w2, cy, x3, cy, "arr-o")
    s += line(x3 + w3, cy, x4, cy, "arr-o")

    ly = y0 + 168
    s += _box(x0 + pw // 2, ly, 360, 52, "#f0faf0", "#27AE60", "NOT used in hash layer", [
        ["<tspan>no </tspan>", t_var("u", "t"), t_dot(), t_var("e", "i"), "<tspan> GEMM, no </tspan>", t_var("b", "i")],
    ])
    s += plain(x0 + pw // 2, ly + 38, "id from hash table / mix function, weak semantics OK", cls="an", anchor="middle")
    return s


def _swap_arrow(s: str) -> str:
    y = P1_Y + 88
    s += plain(MID_X, P1_Y + 28, "浅层 MoE: 换路由函数", cls="st", anchor="middle")
    s += (
        f'<path d="M{MID_X - 8},{y} L{MID_X + 8},{y} L{MID_X + 2},{y - 5} M{MID_X + 8},{y} L{MID_X + 2},{y + 5}" '
        f'stroke="#E67E22" stroke-width="1.5" fill="none"/>'
    )
    return s


def _panel_common(s: str) -> str:
    y0, ph = P2_Y, P2_H
    s += _phase_frame(24, y0, 872, ph, "C  After expert id: same EP + shared engine (V3 family)")

    cy = y0 + 72
    nodes = [
        (118, "Expert ids", [["<tspan>top-</tspan>", t_var("K", "r"), "<tspan> or </tspan>", t_var("i", "t")]], "#f8f9fc", "#c5cee0", 112),
        (300, "EP scatter", [["<tspan>dispatch tokens</tspan>"], ["<tspan>to expert ranks</tspan>"]], "#eef4fc", "#4A90D9", 124),
        (490, "Routed FFN", [["<tspan>sparse experts</tspan>"], ["<tspan>FP4 weights (V4)</tspan>"]], "#fff8ee", "#E67E22", 124),
        (680, "Gather+shared", [["<tspan>combine + always-on</tspan>"], ["<tspan>shared path</tspan>"]], "#f0faf0", "#27AE60", 128),
        (848, "MoE out", [[t_var("y", "t")]], "#f8f9fc", "#c5cee0", 84),
    ]
    xs = [n[0] for n in nodes]
    for cx, title, rows, fill, stroke, bw in nodes:
        s += _box(cx, cy, bw, 62, fill, stroke, title, rows)

    for i, (x1, x2) in enumerate(zip(xs[:-1], xs[1:])):
        hw1 = nodes[i][5] // 2
        hw2 = nodes[i + 1][5] // 2
        cls = "arr-g" if x2 > 500 else "arr-b"
        s += line(x1 + hw1, cy, x2 - hw2, cy, cls)

    s += plain(460, y0 + ph - 14, "Hash only replaces A/B routing; C unchanged vs V3 inference stack", cls="st", anchor="middle")
    return s


def gen_hash_moe_routing() -> None:
    s = ""
    s += plain(W // 2, 22, "Gate affinity to Hash query: V3 centroid routing vs V4 shallow Hash MoE", cls="t", anchor="middle")
    s += plain(
        W // 2,
        40,
        "Swap how expert id is chosen; EP scatter / gather + shared merge stay the same",
        cls="st",
        anchor="middle",
    )
    s = _panel_v3(s)
    s = _panel_v4(s)
    s = _swap_arrow(s)
    s = _panel_common(s)

    for out_dir in (OUT, FIG_V4):
        _write_svg("hash-moe-routing.svg", s, W, H, out_dir)


if __name__ == "__main__":
    gen_hash_moe_routing()
