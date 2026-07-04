#!/usr/bin/env python3
"""MTP intermediate-token fusion scheme (V3 Eq.21-23). All math via tspan (svg-diagram-math.mdc)."""
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
sys.path.insert(0, str(ROOT))

from gen_deepseek_svgs import box, line, phase  # noqa: E402
from svg_math_helpers import (  # noqa: E402
    LABEL_FONT,
    MATH_FONT,
    default_math_styles,
    math_line,
    plain,
    t_arrow,
    t_concat_norms,
    t_emb,
    t_loss,
    t_loss_mtp,
    t_lambda,
    t_outhead,
    t_rmsnorm,
    t_softmax,
    t_supsub,
    t_trm,
    t_var,
)

OUT = DIAGRAMS
FIG_VERSIONS = REPO / "docs" / "versions" / "figures"
PF = "mtpf"
W, H = 780, 840

MAIN_FILL, MAIN_STROKE = "#eef4fc", "#4A90D9"
FUSE_FILL, FUSE_STROKE = "#fef3c7", "#d97706"
MTP_FILL, MTP_STROKE = "#f0faf0", "#27AE60"
TOK_FILL, TOK_STROKE = "#fff8ee", "#E67E22"
HEAD_FILL, HEAD_STROKE = "#ede9fe", "#7c3aed"


def _markers() -> str:
    return f"""
  <marker id="{PF}a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#666"/></marker>
  <marker id="{PF}ad" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#888"/></marker>
  <marker id="{PF}ab" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#4A90D9"/></marker>
  <marker id="{PF}ag" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#27AE60"/></marker>
"""


def _styles() -> str:
    return default_math_styles() + f"""
  .ph {{ font: 600 12px sans-serif; fill: #4a5a7a; }}
  .arr {{ stroke: #666; stroke-width: 1.5; fill: none; marker-end: url(#{PF}a); }}
  .arr-d {{ stroke: #888; stroke-width: 1.2; fill: none; stroke-dasharray: 5 4; marker-end: url(#{PF}ad); }}
  .arr-b {{ stroke: #4A90D9; stroke-width: 1.5; fill: none; marker-end: url(#{PF}ab); }}
  .arr-g {{ stroke: #27AE60; stroke-width: 1.5; fill: none; marker-end: url(#{PF}ag); }}
"""


def _math_text(cx: int, cy: int, parts: list[str], *, cls: str = "m", size: str | None = None) -> str:
    sz = f' font-size="{size}"' if size else ""
    return f'<text x="{cx}" y="{cy}" class="{cls}" text-anchor="middle"{sz}>{"".join(parts)}</text>'


def _box_math(cx: int, cy: int, w: int, h: int, fill: str, stroke: str, parts: list[str], note: str = "") -> str:
    x, y = cx - w // 2, cy - h // 2
    s = box(x, y, w, h, fill, stroke)
    s += _math_text(cx, cy + (-6 if note else 2), parts)
    if note:
        s += plain(cx, cy + 14, note, cls="dt", anchor="middle")
    return s


def _fuse_block(cx: int, cy: int, k: int) -> str:
    w, h = 268, 96
    x, y = cx - w // 2, cy - h // 2
    s = box(x, y, w, h, FUSE_FILL, FUSE_STROKE)
    s += plain(cx, y + 16, "Fusion", cls="lb", anchor="middle")
    s += _math_text(cx, y + 32, [t_var("M", str(k))])
    s += _math_text(
        cx,
        y + 50,
        [t_concat_norms(t_supsub("h", "t", sup=f"({k - 1})"), t_emb(f"t+{k}"))],
        cls="fm",
        size="9px",
    )
    s += _math_text(
        cx,
        y + 66,
        [
            "<tspan>[</tspan>",
            t_concat_norms(t_supsub("h", "t", sup=f"({k - 1})"), t_emb(f"t+{k}")),
            "<tspan>]</tspan>",
            t_arrow(),
            t_var("M", str(k)),
        ],
        cls="fm",
        size="9px",
    )
    s += _math_text(
        cx,
        y + 82,
        [t_arrow(), t_supsub("h", "t", sup=f"'({k})")],
        cls="fm",
        size="9px",
    )
    return s


def gen() -> None:
    s = ""
    s += plain(W // 2, 22, "MTP 中间 token 融合方案（V3 Eq.21-23）", cls="t", anchor="middle")

    s += math_line(
        W // 2,
        48,
        [
            t_supsub("h", "t", sup="'(k)"),
            "<tspan> = </tspan>",
            t_var("M", "k"),
            "<tspan> [ </tspan>",
            t_concat_norms(t_supsub("h", "t", sup="(k-1)"), t_emb("t+k")),
            "<tspan> ]</tspan>",
        ],
    )

    # --- A ---
    s += phase(20, 64, W - 40, 178, "A  单步 decode（位置 t）：主网只跑 1 遍")
    s += _math_text(
        W // 2,
        88,
        [
            "<tspan>...</tspan>",
            t_var("x", "1"),
            "<tspan> ... </tspan>",
            t_var("x", "t"),
            "<tspan>  （已接受前缀）</tspan>",
        ],
        size="10px",
    )

    y = 138
    s += _box_math(
        95,
        y,
        118,
        52,
        MAIN_FILL,
        MAIN_STROKE,
        [
            "<tspan>Main Transformer</tspan>",
            "<tspan> (</tspan>",
            t_var("L", "", italic=True),
            "<tspan> layers)</tspan>",
        ],
    )
    s += _box_math(235, y, 118, 52, MAIN_FILL, MAIN_STROKE, [t_supsub("h", "t", sup="(0)")], "主网 hidden")
    s += _box_math(365, y, 100, 48, HEAD_FILL, HEAD_STROKE, [t_outhead(), "<tspan> (shared)</tspan>"])
    s += _box_math(
        500,
        y,
        118,
        52,
        MAIN_FILL,
        MAIN_STROKE,
        [t_var("x", "t+1"), "<tspan>, </tspan>", t_loss("main")],
    )

    s += line(154, y, 176, y, "arr-b")
    s += line(294, y, 315, y, "arr-b")
    s += line(415, y, 441, y, "arr-b")

    s += _math_text(
        W // 2,
        208,
        [
            "<tspan>主网产出 </tspan>",
            t_supsub("h", "t", sup="(0)"),
            "<tspan> 后，MTP 不再重跑 </tspan>",
            t_var("L", "", italic=True),
            "<tspan> 层主 Transformer</tspan>",
        ],
        cls="an",
        size="9px",
    )
    s += _math_text(
        W // 2,
        224,
        [
            "<tspan>每深度仅 </tspan>",
            t_trm("k"),
            "<tspan>（1 Block）+ </tspan>",
            t_var("M", "k"),
        ],
        cls="an",
        size="9px",
    )

    # --- B ---
    s += phase(20, 254, W - 40, 302, "B  MTP 因果链：逐层融合 hidden + 中间 token embed")

    chain_y = 336
    xs = [82, 286, 468, 568, 688]
    emb_dy = 78
    fuse_hw = 134

    s += _box_math(xs[0], chain_y - emb_dy, 108, 44, TOK_FILL, TOK_STROKE, [t_emb("t+1"), "<tspan> shared</tspan>"])
    s += _fuse_block(xs[1], chain_y, 1)
    s += _box_math(xs[2], chain_y, 96, 48, MTP_FILL, MTP_STROKE, [t_trm("1")])
    s += _box_math(xs[3], chain_y, 88, 44, MTP_FILL, MTP_STROKE, [t_supsub("h", "t", sup="(1)")])
    s += _box_math(
        xs[4],
        chain_y,
        108,
        44,
        HEAD_FILL,
        HEAD_STROKE,
        [t_arrow(), t_var("x", "t+2")],
    )
    s += line(xs[1] + fuse_hw, chain_y, xs[2] - 48, chain_y)
    s += line(xs[2] + 48, chain_y, xs[3] - 44, chain_y)
    s += line(xs[3] + 44, chain_y, xs[4] - 54, chain_y, "arr-g")

    fuse_top = chain_y - 48
    route_y = fuse_top - 18
    s += line(235, y + 26, 235, route_y, "arr-d")
    s += line(235, route_y, xs[1], route_y, "arr-d")
    s += line(xs[1], route_y, xs[1], fuse_top, "arr-d")

    s += line(xs[0] + 54, chain_y - emb_dy + 22, xs[1] - fuse_hw + 8, chain_y - 20, "arr-d")

    chain_y2 = 432
    s += _box_math(xs[0], chain_y2 - emb_dy, 108, 44, TOK_FILL, TOK_STROKE, [t_emb("t+2"), "<tspan> shared</tspan>"])
    s += _fuse_block(xs[1], chain_y2, 2)
    s += _box_math(xs[2], chain_y2, 96, 48, MTP_FILL, MTP_STROKE, [t_trm("2")])
    s += _box_math(xs[3], chain_y2, 88, 44, MTP_FILL, MTP_STROKE, [t_supsub("h", "t", sup="(2)")])
    s += _box_math(
        xs[4],
        chain_y2,
        108,
        44,
        HEAD_FILL,
        HEAD_STROKE,
        [t_arrow(), t_var("x", "t+3")],
    )
    s += line(xs[1] + fuse_hw, chain_y2, xs[2] - 48, chain_y2)
    s += line(xs[2] + 48, chain_y2, xs[3] - 44, chain_y2)
    s += line(xs[3] + 44, chain_y2, xs[4] - 54, chain_y2, "arr-g")
    s += line(xs[3], chain_y + 22, xs[3], chain_y2 - 48, "arr-g")

    s += line(xs[0] + 54, chain_y2 - emb_dy + 22, xs[1] - fuse_hw + 8, chain_y2 - 20, "arr-d")

    s += math_line(
        W // 2,
        512,
        [
            t_supsub("h", "t", sup="(k)"),
            "<tspan> = </tspan>",
            t_trm("k"),
            "<tspan>(</tspan>",
            t_supsub("h", "t", sup="'"),
            "<tspan>)</tspan>",
            t_arrow(),
            t_var("P", "t+k+1"),
            "<tspan> = </tspan>",
            t_softmax("".join([t_outhead(), "<tspan>(</tspan>", t_supsub("h", "t", sup="(k)"), "<tspan>)</tspan>"])),
        ],
        cls="fm",
    )

    s += _math_text(
        W // 2,
        532,
        [
            "<tspan>NOT </tspan>",
            t_softmax("<tspan>...</tspan>"),
            "<tspan> 无依赖并行 </tspan>",
            t_var("x", "t+1"),
            "<tspan>...</tspan>",
            t_var("x", "t+K"),
            "<tspan>；NOT 主网跑 </tspan>",
            t_var("K", "", italic=True),
            "<tspan> 遍</tspan>",
        ],
        cls="st",
        size="9px",
    )

    # --- C ---
    s += phase(20, 562, W - 40, 250, "C  中间 token 来源（训练 vs 推理 draft）")

    s += box(40, 592, 340, 200, MAIN_FILL, MAIN_STROKE)
    s += plain(210, 610, "训练", cls="lb", anchor="middle")
    s += _math_text(
        210,
        632,
        [
            "<tspan>主网 1 次前向 </tspan>",
            t_arrow(),
            t_supsub("h", "t", sup="(0)"),
            "<tspan> 全体 </tspan>",
            t_var("t", "", italic=True),
        ],
        size="10px",
    )
    s += _math_text(
        210,
        652,
        [
            "<tspan>中间 token = teacher forcing </tspan>",
            t_var("x", "t+1"),
            "<tspan>, </tspan>",
            t_var("x", "t+2"),
            "<tspan>, ...</tspan>",
        ],
        size="10px",
    )
    s += _math_text(
        210,
        674,
        [
            t_loss("total"),
            "<tspan> = </tspan>",
            t_loss("main"),
            "<tspan> + </tspan>",
            "<tspan>&#x2211;</tspan>",
            t_lambda("k"),
            t_loss_mtp("k"),
        ],
        size="10px",
    )

    s += box(400, 592, 340, 200, MTP_FILL, MTP_STROKE)
    s += plain(570, 610, "推理（MTP 当 draft）", cls="lb", anchor="middle")
    s += _math_text(
        570,
        632,
        [
            "<tspan>每轮：主网 1 次 verify + MTP 链 </tspan>",
            t_var("K", "", italic=True),
            "<tspan> 小步</tspan>",
        ],
        size="10px",
    )
    s += _math_text(
        570,
        652,
        [
            "<tspan>中间 token = 上一步 </tspan>",
            t_var("x", "t+k", italic=True),
            "<tspan>（</tspan>",
            t_emb("t+k"),
            "<tspan>）</tspan>",
        ],
        size="10px",
    )
    s += _math_text(
        570,
        674,
        [
            "<tspan>draft </tspan>",
            t_arrow(),
            "<tspan> verify </tspan>",
            t_arrow(),
            "<tspan> accept（lossless）</tspan>",
        ],
        size="10px",
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
        p = out_dir / "mtp-fusion-scheme.svg"
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
