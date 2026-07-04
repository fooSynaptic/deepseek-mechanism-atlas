#!/usr/bin/env python3
"""Hyper-Connections: n parallel residual streams + pre/post/comb (HC §1.2)."""
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

from gen_deepseek_svgs import box, line, phase  # noqa: E402
from svg_math_helpers import (  # noqa: E402
    default_math_styles,
    math_line,
    plain,
    t_supsub,
    t_var,
)

OUT = str(DIAGRAMS)
FIG_MHC = str(FIGURES / "mhc")
PF = "hc"

W, H = 920, 420
N = 4
SW, SH, SGAP = 88, 32, 6

STREAM_FILL, STREAM_STROKE = "#eef4fc", "#4A90D9"
MIX_FILL, MIX_STROKE = "#fef3c7", "#d97706"
SUB_FILL, SUB_STROKE = "#f0faf0", "#27AE60"


def _styles() -> str:
    return default_math_styles() + f"""
  .ph {{ font: 600 12px sans-serif; fill: #4a5a7a; }}
  .arr {{ stroke: #666; stroke-width: 1.5; fill: none; marker-end: url(#{PF}a); }}
  .arr-d {{ stroke: #888; stroke-width: 1.2; fill: none; stroke-dasharray: 5 4; marker-end: url(#{PF}ad); }}
  .arr-b {{ stroke: #4A90D9; stroke-width: 1.5; fill: none; marker-end: url(#{PF}ab); }}
  .arr-o {{ stroke: #E67E22; stroke-width: 1.5; fill: none; marker-end: url(#{PF}ao); }}
  .arr-g {{ stroke: #27AE60; stroke-width: 1.5; fill: none; marker-end: url(#{PF}ag); }}
"""


def _header(w: int, h: int) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
<style><![CDATA[
{_styles()}
]]></style>
<defs>
  <marker id="{PF}a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#666"/></marker>
  <marker id="{PF}ad" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#888"/></marker>
  <marker id="{PF}ab" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#4A90D9"/></marker>
  <marker id="{PF}ao" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#E67E22"/></marker>
  <marker id="{PF}ag" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#27AE60"/></marker>
</defs>
'''


def _stream_stack(x: int, y0: int, *, layer_tag: str, out: bool) -> tuple[str, list[tuple[int, int]]]:
    """Vertical n stream boxes; return svg + right-edge centers."""
    s: list[str] = []
    centers: list[tuple[int, int]] = []
    for i in range(N):
        y = y0 + i * (SH + SGAP)
        cx = x + SW // 2
        cy = y + SH // 2
        centers.append((x + SW, cy))
        s.append(box(x, y, SW, SH, STREAM_FILL, STREAM_STROKE))
        idx = i + 1
        if layer_tag == "l":
            parts = [t_supsub("r", "l", sup=f",{idx}")]
        else:
            parts = [t_supsub("r", "l+1", sup=f",{idx}")]
        s.append(
            f'<text x="{cx}" y="{cy + 5}" class="fm" text-anchor="middle">'
            f'{"".join(parts)}</text>'
        )
    tag = f"{layer_tag}+1" if out else layer_tag
    s.append(plain(x + SW // 2, y0 - 10, f"n={N} streams @ layer {tag}", cls="an", anchor="middle"))
    return "\n".join(s), centers


def _mix_box(cx: int, cy: int, w: int, h: int, title: str, parts: list[str]) -> str:
    x, y = cx - w // 2, cy - h // 2
    s = box(x, y, w, h, MIX_FILL, MIX_STROKE)
    s += plain(cx, y + 16, title, cls="lb", anchor="middle")
    s += f'<text x="{cx}" y="{y + 38}" class="fm" text-anchor="middle">{"".join(parts)}</text>'
    return s


def _standard_residual() -> str:
    y = 78
    s = phase(28, 52, 864, 92, "A  Standard residual (single stream)")
    boxes = [
        (120, "x_l", "#f8f9fc"),
        (310, "F_l", SUB_FILL),
        (500, "x_{l+1}", "#f8f9fc"),
    ]
    for cx, label, fill in boxes:
        bh = 52 if label == "F_l" else 40
        s += box(cx - 54, y, 108, bh, fill, "#c5cee0")
        if label == "x_{l+1}":
            s += math_line(cx, y + 22, [t_supsub("x", "l+1")])
        elif label == "F_l":
            s += plain(cx, y + 22, "F_l", cls="lb", anchor="middle")
            s += plain(cx, y + 38, "Attn / FFN", cls="dt", anchor="middle")
        else:
            s += math_line(cx, y + 22, [t_supsub("x", "l")])
    mid = y + 26
    s += line(174, mid, 256, mid, "arr-b")
    s += line(364, mid, 446, mid, "arr-b")
    s += f'<path d="M120,{mid} Q120,{y - 8} 446,{y - 8}" class="arr-d" fill="none" marker-end="url(#{PF}ad)"/>'
    s += plain(310, y - 18, "identity skip (+ x_l)", cls="an", anchor="middle")
    return s


def _hc_layer() -> str:
    phase_y, phase_h = 156, 232
    stack_h = N * SH + (N - 1) * SGAP
    y0 = phase_y + 36 + (phase_h - 36 - stack_h) // 2
    lx, rx = 72, 720
    s = phase(28, phase_y, 864, phase_h, "B  Hyper-Connections (one sublayer; n=4)")

    left_svg, left_pts = _stream_stack(lx, y0, layer_tag="l", out=False)
    right_svg, _ = _stream_stack(rx, y0, layer_tag="l", out=True)
    s += left_svg + right_svg

    mid_y = y0 + stack_h // 2
    pre_x, sub_x, post_x = 280, 430, 580

    s += _mix_box(pre_x, mid_y, 96, 52, "Pre-mix", [t_var("H", "pre")])
    s += box(sub_x - 62, mid_y - 30, 124, 60, SUB_FILL, SUB_STROKE)
    s += plain(sub_x, mid_y - 6, "F_l", cls="lb", anchor="middle")
    s += plain(sub_x, mid_y + 10, "Attn / FFN", cls="dt", anchor="middle")
    s += math_line(sub_x, mid_y + 26, [t_supsub("y", "l")])

    s += box(post_x - 70, mid_y - 34, 140, 68, MIX_FILL, MIX_STROKE)
    s += plain(post_x, mid_y - 14, "Post-mix", cls="lb", anchor="middle")
    s += f'<text x="{post_x}" y="{mid_y + 8}" class="fm" text-anchor="middle">{"".join([t_var("H", "post")])}</text>'
    s += f'<text x="{post_x}" y="{mid_y + 24}" class="fm" text-anchor="middle">{"".join([t_var("H", "comb")])}</text>'

    for _, cy in left_pts:
        s += line(lx + SW, cy, pre_x - 48, mid_y, "arr-o")
    s += line(pre_x + 48, mid_y, sub_x - 62, mid_y, "arr-b")
    s += line(sub_x + 62, mid_y, post_x - 70, mid_y, "arr-b")

    right_centers = [
        (rx, y0 + i * (SH + SGAP) + SH // 2) for i in range(N)
    ]
    for cx, cy in right_centers:
        s += line(post_x + 70, mid_y, rx, cy, "arr-o")

    s += plain(430, phase_y + phase_h + 14, "NOT replacing Attn/FFN internals; only residual topology widens to n parallel highways", cls="st", anchor="middle")
    return s


def gen() -> None:
    body = [
        plain(W // 2, 24, "Hyper-Connections (HC): residual topology", cls="t", anchor="middle"),
        plain(W // 2, 42, "single stream x_l  vs  n parallel residual streams + H^pre / H^post / H^comb", cls="st", anchor="middle"),
        _standard_residual(),
        _hc_layer(),
    ]
    content = _header(W, H) + "\n".join(body) + "\n</svg>"

    for out_dir in (OUT, FIG_MHC):
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, "hyper-connections.svg")
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
