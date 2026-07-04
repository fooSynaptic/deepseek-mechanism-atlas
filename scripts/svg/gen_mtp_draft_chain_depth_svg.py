#!/usr/bin/env python3
"""MTP draft chain: serial depth grows; shared OutHead (not K separate LM heads)."""
from __future__ import annotations

from pathlib import Path

import sys

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DIAGRAMS, FIGURES, REPO  # noqa: E402
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))

from gen_deepseek_svgs import box, line, phase  # noqa: E402
from svg_math_helpers import (  # noqa: E402
    LABEL_FONT,
    MATH_FONT,
    default_math_styles,
    math_line,
    plain,
    t_emb,
    t_supsub,
    t_trm,
    t_var,
)

OUT = DIAGRAMS
FIG_VERSIONS = REPO / "docs" / "versions" / "figures"
PF = "mtpc"
W, H = 920, 548
PANEL_Y, PANEL_H = 52, 478

MAIN_FILL, MAIN_STROKE = "#eef4fc", "#4A90D9"
FUSE_FILL, FUSE_STROKE = "#fef3c7", "#d97706"
MTP_FILL, MTP_STROKE = "#f0faf0", "#27AE60"
TOK_FILL, TOK_STROKE = "#fff8ee", "#E67E22"
HEAD_FILL, HEAD_STROKE = "#ede9fe", "#7c3aed"
CHAIN_FILL, CHAIN_STROKE = "#f1f5f9", "#94a3b8"


def _markers() -> str:
    return f"""
  <marker id="{PF}a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#475569"/></marker>
  <marker id="{PF}ab" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#4A90D9"/></marker>
  <marker id="{PF}ag" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#27AE60"/></marker>
  <marker id="{PF}ad" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#888"/></marker>
"""


def _styles() -> str:
    return f"""
{default_math_styles()}
  .ph {{ font: 600 12px sans-serif; fill: #4a5a7a; }}
  .arr {{ stroke: #475569; stroke-width: 1.5; fill: none; marker-end: url(#{PF}a); }}
  .arr-b {{ stroke: #4A90D9; stroke-width: 1.5; fill: none; marker-end: url(#{PF}ab); }}
  .arr-g {{ stroke: #27AE60; stroke-width: 1.5; fill: none; marker-end: url(#{PF}ag); }}
  .arr-d {{ stroke: #888; stroke-width: 1.2; fill: none; stroke-dasharray: 5 4; marker-end: url(#{PF}ad); }}
  .m, .m tspan {{ font-family: {MATH_FONT}; }}
  .depth {{ font-size: 9px; fill: #64748b; font-family: {LABEL_FONT}; }}
  .sn {{ font-family: {LABEL_FONT}; font-size: 9px; fill: #64748b; }}
  .sn .m {{ font-family: {MATH_FONT}; fill: #64748b; }}
  .em {{ font-weight: 700; }}
"""


def _node(cx: int, cy: int, w: int, h: int, fill: str, stroke: str, math: str, sub: str = "") -> str:
    x, y = cx - w // 2, cy - h // 2
    s = box(x, y, w, h, fill, stroke)
    s += f'<text x="{cx}" y="{cy + (4 if not sub else -2)}" class="m" text-anchor="middle">{math}</text>'
    if sub:
        s += plain(cx, cy + 14, sub, cls="depth", anchor="middle")
    return s


def _outhead(cx: int, cy: int) -> str:
    return _node(cx, cy, 88, 40, HEAD_FILL, HEAD_STROKE, "<tspan>OutHead</tspan>", "shared")


def gen() -> None:
    s = ""
    s += plain(W // 2, 22, "MTP draft 计算链：串行、非一次性；链深度递增", cls="t", anchor="middle")
    s += math_line(
        W // 2,
        40,
        [
            '<tspan class="sn">K 次预测共用 1 个 OutHead；第 </tspan>',
            t_var("k", "", italic=True),
            '<tspan class="sn"> 步多融 1 个 </tspan>',
            t_emb("t+k"),
            '<tspan class="sn">，路径经过 </tspan>',
            t_var("k", "", italic=True),
            '<tspan class="sn"> 个 Fusion+TRM</tspan>',
        ],
        cls="s",
    )

    s += phase(24, PANEL_Y, 872, PANEL_H, "细节计算图（位置 t；推理 draft 时 xt+k 为上一步 propose）")

    # Row labels: chain depth
    depths = [
        (0, "链深度 0", "0 个中间 Emb"),
        (1, "链深度 1", "1 个 Emb"),
        (2, "链深度 2", "2 个 Emb"),
    ]
    ys = [118, 228, 338]
    for (d, title, note), y in zip(depths, ys):
        s += plain(72, y, title, cls="lb", anchor="middle")
        s += plain(72, y + 14, note, cls="depth", anchor="middle")

    # --- depth 0: main -> x_{t+1} ---
    y0 = ys[0]
    s += _node(220, y0, 100, 44, MAIN_FILL, MAIN_STROKE, t_supsub("h", "t", sup="(0)"))
    s += _outhead(360, y0)
    s += _node(500, y0, 80, 44, MAIN_FILL, MAIN_STROKE, t_var("x", "t+1"), "propose #1")
    s += line(270, y0, 316, y0, "arr-b")
    s += line(404, y0, 460, y0, "arr-b")

    # --- depth 1: + Emb(x_{t+1}) + M1 + TRM1 -> x_{t+2} ---
    y1 = ys[1]
    s += _node(220, y1, 100, 44, MAIN_FILL, MAIN_STROKE, t_supsub("h", "t", sup="(0)"))
    s += _node(330, y1, 72, 40, TOK_FILL, TOK_STROKE, t_emb("t+1"), "")
    s += _node(430, y1, 72, 44, FUSE_FILL, FUSE_STROKE, t_var("M", "1"), "Fusion")
    s += _node(520, y1, 72, 44, MTP_FILL, MTP_STROKE, t_trm("1"), "")
    s += _node(610, y1, 88, 44, MTP_FILL, MTP_STROKE, t_supsub("h", "t", sup="(1)"))
    s += _outhead(710, y1)
    s += _node(820, y1, 80, 44, MTP_FILL, MTP_STROKE, t_var("x", "t+2"), "propose #2")

    s += line(270, y1, 296, y1, "arr-b")
    s += line(366, y1, 394, y1)
    s += line(466, y1, 484, y1)
    s += line(556, y1, 566, y1)
    s += line(654, y1, 666, y1, "arr-g")
    s += line(754, y1, 780, y1, "arr-g")
    s += line(330, y1 - 20, 330, y1 - 50, "arr-d")
    s += line(330, y1 - 50, 500, y1 - 50, "arr-d")
    s += line(500, y1 - 50, 500, y0 + 22, "arr-d")
    s += plain(415, y1 - 58, "复用上一步 propose 的 token embed", cls="depth", anchor="middle")

    # vertical chain link h^(0) down from row0
    s += line(220, y0 + 22, 220, y1 - 22, "arr-d")

    # --- depth 2: h^(1) + Emb(x_{t+2}) + M2 + TRM2 -> x_{t+3} ---
    y2 = ys[2]
    s += _node(220, y2, 100, 44, MTP_FILL, MTP_STROKE, t_supsub("h", "t", sup="(1)"))
    s += _node(330, y2, 72, 40, TOK_FILL, TOK_STROKE, t_emb("t+2"), "")
    s += _node(430, y2, 72, 44, FUSE_FILL, FUSE_STROKE, t_var("M", "2"), "Fusion")
    s += _node(520, y2, 72, 44, MTP_FILL, MTP_STROKE, t_trm("2"), "")
    s += _node(610, y2, 88, 44, MTP_FILL, MTP_STROKE, t_supsub("h", "t", sup="(2)"))
    s += _outhead(710, y2)
    s += _node(820, y2, 80, 44, MTP_FILL, MTP_STROKE, t_var("x", "t+3"), "propose #3")

    s += line(270, y2, 296, y2, "arr-g")
    s += line(366, y2, 394, y2)
    s += line(466, y2, 484, y2)
    s += line(556, y2, 566, y2)
    s += line(654, y2, 666, y2, "arr-g")
    s += line(754, y2, 780, y2, "arr-g")
    s += line(610, y1 + 22, 610, y2 - 22, "arr-g")
    s += line(330, y2 - 20, 330, y2 - 48, "arr-d")
    s += line(330, y2 - 48, 820, y2 - 48, "arr-d")
    s += line(820, y2 - 48, 820, y1 + 22, "arr-d")

    # shared OutHead annotation
    s += box(680, 88, 160, 36, HEAD_FILL, HEAD_STROKE)
    s += plain(760, 102, "OutHead (shared)", cls="lb", anchor="middle")
    s += plain(760, 116, "不是 K 个独立 LM 头", cls="depth", anchor="middle")
    for y in ys:
        s += line(760, 124, 710, y, "arr-d")

    note_y1 = PANEL_Y + PANEL_H - 44
    note_y2 = PANEL_Y + PANEL_H - 24
    s += plain(
        W // 2,
        note_y1,
        "NOT 一次性 K 路并行 softmax；每多猜 1 个 token = 多 1 步 Fusion+TRM + 多 1 次 Emb 注入",
        cls="st",
        anchor="middle",
    )
    s += (
        f'<text x="{W // 2}" y="{note_y2}" class="m" text-anchor="middle" font-size="10px">'
        f"<tspan>第 </tspan>{t_var('k', '', italic=False)}"
        f"<tspan> 个 MTP 步预测 </tspan>{t_var('x', 't+k+1')}"
        f"<tspan>；TRM</tspan>"
        f'<tspan baseline-shift="sub" font-size="0.72em">k</tspan>'
        f"<tspan> 仍只有 </tspan>"
        f'<tspan class="em">1</tspan>'
        f"<tspan> Block，变长的是 </tspan>"
        f'<tspan class="em">链</tspan>'
        f"<tspan> 而非单层堆深</tspan></text>\n"
    )

    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<style><![CDATA[
{_styles()}
]]></style>
<defs>
{_markers()}
</defs>
{s}
</svg>"""

    for out_dir in (OUT, FIG_VERSIONS):
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "mtp-draft-chain-depth.svg"
        p.write_text(content, encoding="utf-8")
        ET.parse(p)
        sys.path.insert(0, str(ROOT))
        from svg_validate import validate_svg  # noqa: E402

        errs = validate_svg(p)
        if errs:
            raise SystemExit(f"FAIL {p.name}: {', '.join(errs)}")
        print("OK", p)


if __name__ == "__main__":
    gen()
