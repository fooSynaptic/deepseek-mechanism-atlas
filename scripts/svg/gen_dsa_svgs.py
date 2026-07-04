#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate SVG diagrams for deepseek-ai/dsa."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DSA_DIAGRAMS, REPO  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))

import html
import os
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = str(DSA_DIAGRAMS)





from svg_math_helpers import (  # noqa: E402
    blt_math,
    default_math_styles,
    math_line,
    plain,
    t_arrow,
    t_c,
    t_cn,
    t_dot,
    t_in,
    t_idx,
    t_loss,
    t_relu,
    t_rope,
    t_srow,
    t_softmax,
    t_sum,
    t_supsub,
    t_times,
    t_var,
    t_w,
)


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def svg_header(w: int, h: int, prefix: str) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
<style><![CDATA[
  .t {{ font: 700 15px sans-serif; fill: #1a1a2e; text-anchor: middle; }}
  .st {{ font: 11px sans-serif; fill: #666; text-anchor: middle; }}
  .lb {{ font: 600 11px sans-serif; fill: #222; text-anchor: middle; }}
  .dt {{ font: 10px sans-serif; fill: #555; text-anchor: middle; }}
  .an {{ font: 9px sans-serif; fill: #2563eb; text-anchor: middle; }}
  .bl {{ font: 10px sans-serif; fill: #555; text-anchor: start; }}
  .arr {{ stroke: #666; stroke-width: 1.5; fill: none; marker-end: url(#{prefix}a); }}
  .arr-b {{ stroke: #4A90D9; stroke-width: 1.5; fill: none; marker-end: url(#{prefix}ab); }}
  .arr-g {{ stroke: #27AE60; stroke-width: 1.5; fill: none; marker-end: url(#{prefix}ag); }}
]]></style>
<defs>
  <marker id="{prefix}a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#666"/></marker>
  <marker id="{prefix}ab" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#4A90D9"/></marker>
  <marker id="{prefix}ag" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#27AE60"/></marker>
</defs>
'''


def box(x, y, w, h, fill="#f8f9fc", stroke="#c5cee0", rx=6):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
    )


def txt(x, y, s, cls="lb"):
    return f'<text x="{x}" y="{y}" class="{cls}">{esc(s)}</text>'


def line(x1, y1, x2, y2, cls="arr"):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}"/>'


def node(cx, cy, w, h, title, lines, fill="#f8f9fc", stroke="#c5cee0"):
    x, y = cx - w // 2, cy - h // 2
    s = box(x, y, w, h, fill, stroke)
    s += txt(cx, cy - 10, title)
    for i, ln in enumerate(lines):
        s += txt(cx, cy + 8 + i * 14, ln, "dt")
    return s


def svg_header_math(w: int, h: int, prefix: str) -> str:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
<style><![CDATA[
{default_math_styles()}
  .arr {{ stroke: #666; stroke-width: 1.5; fill: none; marker-end: url(#{prefix}a); }}
  .arr-b {{ stroke: #4A90D9; stroke-width: 1.5; fill: none; marker-end: url(#{prefix}ab); }}
  .arr-g {{ stroke: #27AE60; stroke-width: 1.5; fill: none; marker-end: url(#{prefix}ag); }}
]]></style>
<defs>
  <marker id="{prefix}a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#666"/></marker>
  <marker id="{prefix}ab" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#4A90D9"/></marker>
  <marker id="{prefix}ag" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#27AE60"/></marker>
</defs>
'''


def math_txt(
    cx: int,
    y: int,
    parts: list[str],
    *,
    cls: str = "fm",
    anchor: str = "middle",
    size: str | None = None,
) -> str:
    sz = f' font-size="{size}"' if size else ""
    return f'<text x="{cx}" y="{y}" class="{cls}" text-anchor="{anchor}"{sz}>{"".join(parts)}</text>'


def write(name: str, body: str, w: int, h: int, prefix: str, *, math: bool = False) -> None:
    header = svg_header_math(w, h, prefix) if math else svg_header(w, h, prefix)
    content = header + body + "</svg>"
    p = os.path.join(OUT, name)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    ET.parse(p)
    print("OK", name)
    # 布局 + 结构校验（生成器产出必过）
    import sys

    sys.path.insert(0, os.path.join(OUT, "..", "..", "diagrams"))
    from svg_validate import validate_svg  # noqa: E402

    layout_errs = validate_svg(Path(p))
    if layout_errs:
        raise SystemExit(f"FAIL {name}: {', '.join(layout_errs)}")
    # docs/versions 引用副本
    if name == "ess-dual-cache.svg":
        fig = os.path.join(OUT, "..", "..", "docs", "figures", "ess", name)
        os.makedirs(os.path.dirname(fig), exist_ok=True)
        with open(fig, "w", encoding="utf-8") as f:
            f.write(content)
        print("OK", "docs/figures/ess/" + name)


def gen_dsa_pipeline() -> None:
    w, h = 920, 320
    s = ""
    s += txt(460, 22, "DSA 两阶段稀疏注意力与异构 Cache", "t")
    s += txt(460, 40, "Indexer O(L^2) 轻量全扫 -> Top-k=2048 -> Core MLA O(Lk)", "st")

    y = 100
    s += node(120, y, 130, 72, "Query h_t", ["当前 token"], "#eef4fc", "#4A90D9")
    s += node(300, y, 150, 72, "Lightning Indexer", ["廉价点积打分", "O(L^2) 小常数"], "#fff8ee", "#E67E22")
    s += node(500, y, 130, 72, "Top-k", ["k = 2048", "index 集 I"], "#f0faf0", "#27AE60")
    s += node(720, y, 150, 72, "Core MLA Attn", ["仅 I 内 latent", "O(Lk)"], "#eef4fc", "#4A90D9")

    s += line(185, y, 225, y)
    s += line(375, y, 435, y)
    s += line(565, y, 645, y)

    cy = 230
    s += box(60, cy - 40, 200, 80, "#fff0f0", "#e8a0a0")
    s += txt(160, cy - 22, "Indexer-Cache", "lb")
    s += txt(160, cy - 4, "~16.8%  GPU 常驻", "dt")
    s += txt(160, cy + 12, "每步全算 indexer", "dt")

    s += box(360, cy - 40, 200, 80, "#f0f4ff", "#a0b8e8")
    s += txt(460, cy - 22, "Latent-Cache", "lb")
    s += txt(460, cy - 4, "~83.2%  可 ESS offload", "dt")
    s += txt(460, cy + 12, "仅 top-k entry 进主 attn", "dt")

    s += box(660, cy - 40, 200, 80, "#f0faf0", "#8fbc8f")
    s += txt(760, cy - 22, "Index Share", "lb")
    s += txt(760, cy - 4, "infra 补丁 (后文)", "dt")
    s += txt(760, cy + 12, "跨层复用 top-k index", "dt")

    s += line(300, y + 36, 160, cy - 40, "arr-b")
    s += line(500, y + 36, 460, cy - 40, "arr-b")
    s += line(720, y + 36, 760, cy - 40, "arr-g")
    s += txt(400, 290, "双 Cache 是 Index Share 与 ESS 的结构基础", "an")

    write("dsa-pipeline.svg", s, w, h, "dp")


def gen_index_share_fffs() -> None:
    w, h = 920, 280
    s = ""
    s += txt(460, 22, "Index Share: FFFS 跨层 index 复用", "t")
    s += txt(460, 40, "每 4 层 3 次 indexer + 1 次复用 -> 减 75% indexer 计算", "st")

    bw, bh = 88, 56
    y = 120
    layers = [
        (120, "L1", "F", "算 indexer", "#eef4fc", "#4A90D9"),
        (260, "L2", "F", "算 indexer", "#eef4fc", "#4A90D9"),
        (400, "L3", "F", "算 indexer", "#eef4fc", "#4A90D9"),
        (540, "L4", "S", "复用 L3", "#f0faf0", "#27AE60"),
        (680, "L5", "F", "算 indexer", "#eef4fc", "#4A90D9"),
        (820, "L6", "F", "...", "#f8f9fc", "#c5cee0"),
    ]
    for cx, lid, role, note, fill, stroke in layers:
        s += box(cx - bw // 2, y - bh // 2, bw, bh, fill, stroke)
        s += txt(cx, y - 12, lid, "lb")
        s += txt(cx, y + 4, role, "lb")
        s += txt(cx, y + 20, note, "dt")

    for i in range(len(layers) - 1):
        x1 = layers[i][0] + bw // 2
        x2 = layers[i + 1][0] - bw // 2
        s += line(x1, y, x2, y)

    s += line(400, y - bh // 2, 540, y - bh // 2 - 20, "arr-g")
    s += line(540, y - bh // 2 - 20, 540, y - bh // 2, "arr-g")
    s += txt(470, y - 52, "cached indices", "an")

    s += txt(460, 210, "模式 F F F S 周期重复  |  零额外显存", "st")
    s += txt(460, 248, "仅适用于 DSA 系 (V3.2 / GLM-5)  |  与 ESS offload 正交", "an")

    write("index-share-fffs.svg", s, w, h, "is")


def panel(x, y, w, h, title, fill, stroke):
    s = box(x, y, w, h, fill, stroke)
    if title:
        s += txt(x + w // 2, y + 16, title, "lb")
    return s


def label_bg(x, y, w, h, text, cls="an"):
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="#ffffff" fill-opacity="0.92" stroke="#cbd5e1" stroke-width="1"/>'
    s += txt(x + w // 2, y + h // 2 + 4, text, cls)
    return s


def math_label_bg(x, y, w, h, parts: list[str]) -> str:
    s = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="4" fill="#ffffff" fill-opacity="0.92" stroke="#cbd5e1" stroke-width="1"/>'
    s += math_txt(x + w // 2, y + h // 2 + 4, parts, cls="fm an", size="9px")
    return s


def polyline(points, cls="arr-g"):
    pts = " ".join(f"{x},{y}" for x, y in points)
    return f'<polyline points="{pts}" class="{cls}" fill="none"/>'


def gen_ess_dual_cache() -> None:
    """MLA decode 一步: Indexer vs Latent-Cache 分工 (tspan math, svg-diagram-math.mdc)."""
    w, h = 920, 780
    s = ""
    s += plain(w // 2, 22, "DSA + MLA Decode 一步示例 (token t)", cls="t", anchor="middle")
    s += math_line(
        w // 2,
        40,
        [
            "<tspan class='s' font-family='system-ui,sans-serif' font-size='11px' fill='#666'>例: </tspan>",
            t_var("L", "", italic=True),
            "<tspan class='s' font-family='system-ui,sans-serif' font-size='11px' fill='#666'>=32K | top-</tspan>",
            t_var("k", "", italic=False),
            "<tspan class='s' font-family='system-ui,sans-serif' font-size='11px' fill='#666'>=2048 | </tspan>",
            t_c("KV"),
            "<tspan class='s' font-family='system-ui,sans-serif' font-size='11px' fill='#666'> dim=512 (V2/V3 MLA)</tspan>",
        ],
        cls="s",
    )

    s += box(380, 52, 160, 48, "#1e293b", "#334155")
    s += math_txt(
        460,
        74,
        [t_var("h", "t"), "<tspan> [5120]</tspan>"],
        cls="m",
    )
    s += plain(460, 88, "当前 decode 隐状态", cls="an", anchor="middle")

    qy = 108
    s += box(40, qy, 840, 58, "#eff6ff", "#93c5fd")
    s += plain(460, qy + 18, "MLA Query (每步现算, 不进 KV cache)", cls="lb", anchor="middle")
    s += math_txt(
        280,
        qy + 40,
        [
            t_c("Q"),
            "<tspan> = </tspan>",
            t_w("DQ"),
            t_var("h", "t"),
            "<tspan> [1536]</tspan>",
        ],
    )
    s += math_txt(
        460,
        qy + 40,
        [
            t_var("q", "t,i"),
            "<tspan> = [</tspan>",
            t_supsub("q", "", sup="C"),
            "<tspan>; </tspan>",
            t_supsub("q", "", sup="R"),
            "<tspan>] [128 </tspan>",
            "<tspan>&#xD7;</tspan>",
            "<tspan> 192]</tspan>",
        ],
    )
    s += plain(680, qy + 40, "供 Indexer 打分 + Core Attn", cls="dt", anchor="middle")
    s += line(460, 100, 460, qy, "arr-b")

    py, ph, pw = 180, 408, 430
    lx, rx = 24, 466
    lcx, rcx = lx + pw // 2, rx + pw // 2
    gutter = (lx + pw + rx) // 2

    s += panel(lx, py, pw, ph, "Indexer 做什么", "#fff7ed", "#fdba74")
    s += plain(lcx, py + 32, "Indexer-Cache (GPU 常驻 ~16.8%)", cls="dt", anchor="middle")
    s += plain(lcx, py + 48, "存全长 L 个 indexer 轻量向量 (打分用, 非 MLA 主 KV)", cls="dt", anchor="middle")

    y1 = py + 64
    s += box(lx + 14, y1, pw - 28, 58, "#ffedd5", "#f97316")
    s += plain(lcx, y1 + 18, "Lightning Indexer", cls="lb", anchor="middle")
    s += math_txt(
        lcx,
        y1 + 34,
        [
            "<tspan>s=1..L: </tspan>",
            t_var("I", "t,s"),
            "<tspan> = dot(</tspan>",
            t_var("q", "t"),
            "<tspan>, </tspan>",
            t_var("k", "s"),
            "<tspan>)</tspan>",
        ],
    )
    s += math_txt(
        lcx,
        y1 + 48,
        [
            t_var("O", "L", italic=False),
            "<tspan>; </tspan>",
            t_var("q", "t"),
            "<tspan>=当前 query, </tspan>",
            t_var("k", "s"),
            "<tspan>=历史 token </tspan>",
            t_var("s", "", italic=True),
        ],
        size="9px",
    )

    y2 = y1 + 68
    s += box(lx + 14, y2, pw - 28, 44, "#f0fdf4", "#22c55e")
    s += plain(lcx, y2 + 18, "Top-k 选择", cls="lb", anchor="middle")
    s += math_txt(
        lcx,
        y2 + 34,
        [
            t_var("q", "t"),
            "<tspan> 从 </tspan>",
            t_var("L", "", italic=True),
            "<tspan> 条历史取 top </tspan>",
            t_var("k", "", italic=False),
            "<tspan>=2048 个下标 </tspan>",
            t_var("s", "", italic=True),
        ],
    )

    y3 = y2 + 54
    s += box(lx + 14, y3, pw - 28, 56, "#dcfce7", "#16a34a")
    s += plain(lcx, y3 + 18, "输出 index 集合 I", cls="lb", anchor="middle")
    s += math_txt(
        lcx,
        y3 + 34,
        [
            t_var("I", "", italic=True),
            "<tspan> = {12, 45, 891, ...}  共 2048 个历史下标 </tspan>",
            t_var("s", "", italic=True),
        ],
    )
    s += math_txt(
        lcx,
        y3 + 48,
        [
            "<tspan>只决定 attend 哪些历史 token, 不读 </tspan>",
            t_c("KV"),
        ],
        size="9px",
    )

    y4 = y3 + 66
    s += box(lx + 14, y4, pw - 28, 82, "#fff0f0", "#e8a0a0")
    s += plain(lcx, y4 + 18, "Indexer 不做的事", cls="lb", anchor="middle")
    s += plain(lcx, y4 + 34, "x 不升维 latent  |  x 不算 softmax", cls="dt", anchor="middle")
    s += plain(lcx, y4 + 48, "x 不 offload (每步必须全长扫)", cls="dt", anchor="middle")
    s += math_txt(
        lcx,
        y4 + 62,
        ["<tspan>x 不存 512 维 </tspan>", t_c("KV")],
        size="9px",
    )

    s += panel(rx, py, pw, ph, "Latent-Cache 做什么", "#eff6ff", "#93c5fd")
    s += plain(rcx, py + 32, "Latent-Cache (~83.2%, ESS 可 offload)", cls="dt", anchor="middle")
    s += math_txt(
        rcx,
        py + 48,
        [
            "<tspan>存每历史 </tspan>",
            t_var("s", "", italic=True),
            "<tspan>: </tspan>",
            t_c("KV", "s"),
            "<tspan> [512] + </tspan>",
            t_supsub("k", "", sup="R"),
            "<tspan>; Decode 只读 </tspan>",
            t_var("s", "", italic=True),
            t_in(),
            t_var("I", "", italic=True),
        ],
        size="9px",
    )

    r1 = py + 64
    s += box(rx + 14, r1, pw - 28, 58, "#dbeafe", "#3b82f6")
    s += math_txt(
        rcx,
        r1 + 18,
        [
            "<tspan>ESS Prefetch (仅 </tspan>",
            t_var("I", "", italic=True),
            "<tspan> 内 entry)</tspan>",
        ],
        cls="lb",
    )
    s += plain(rcx, r1 + 34, "GPU 热池命中直接用  |  miss 则 FlashTrans 拉回", cls="dt", anchor="middle")
    s += math_txt(
        rcx,
        r1 + 48,
        [
            "<tspan>656B/entry  |  相邻 step 的 </tspan>",
            t_var("I", "", italic=True),
            "<tspan> 重叠高</tspan>",
        ],
        size="9px",
    )

    r2 = r1 + 68
    s += box(rx + 14, r2, pw - 28, 58, "#bfdbfe", "#2563eb")
    s += math_txt(
        rcx,
        r2 + 18,
        [
            "<tspan>升维 (仅 </tspan>",
            t_var("s", "", italic=True),
            t_in(),
            t_var("I", "", italic=True),
            "<tspan>)</tspan>",
        ],
        cls="lb",
    )
    s += math_txt(
        rcx,
        r2 + 34,
        [
            t_var("K", "s", italic=True),
            "<tspan> = </tspan>",
            t_w("UK"),
            t_c("KV", "s"),
            "<tspan> + </tspan>",
            t_rope(t_supsub("k", "", sup="R")),
            "<tspan>  |  </tspan>",
            t_var("V", "s", italic=True),
            "<tspan> = </tspan>",
            t_w("UV"),
            t_c("KV", "s"),
        ],
        size="9px",
    )
    s += plain(rcx, r2 + 48, "2048 个历史 token x 128 头, 现场展开", cls="dt", anchor="middle")

    r3 = r2 + 68
    s += box(rx + 14, r3, pw - 28, 58, "#eef4fc", "#4A90D9")
    s += plain(rcx, r3 + 18, "Core MLA Attention", cls="lb", anchor="middle")
    s += math_txt(
        rcx,
        r3 + 34,
        [
            t_var("o", "t"),
            "<tspan> = </tspan>",
            "<tspan>&#x2211;</tspan>",
            t_in(),
            t_var("I", "", italic=True),
            "<tspan> </tspan>",
            t_softmax(
                "".join(
                    [
                        t_var("q", "t"),
                        t_var("K", "s", italic=True),
                        "<tspan>^T</tspan>",
                    ]
                )
            ),
            t_var("V", "s", italic=True),
        ],
        size="9px",
    )
    s += math_txt(
        rcx,
        r3 + 48,
        [
            t_var("O", "Lk", italic=False),
            "<tspan>  </tspan>",
            t_var("k", "", italic=False),
            "<tspan>=2048  |  精度敏感主算子</tspan>",
        ],
        size="9px",
    )

    r4 = r3 + 68
    s += box(rx + 14, r4, pw - 28, 64, "#f0f4ff", "#a0b8e8")
    s += plain(rcx, r4 + 18, "输出", cls="lb", anchor="middle")
    s += math_txt(
        rcx,
        r4 + 34,
        [
            t_var("u", "t"),
            "<tspan> = </tspan>",
            t_w("O"),
            "<tspan> concat(</tspan>",
            t_var("o", "t,i"),
            "<tspan>) [5120]</tspan>",
        ],
    )
    s += math_txt(
        rcx,
        r4 + 48,
        [
            "<tspan>本步新 </tspan>",
            t_c("KV", "t"),
            "<tspan> 写回 cache (D2H 可 offload)</tspan>",
        ],
        size="9px",
    )

    q_bottom = qy + 58
    s += polyline([(200, q_bottom), (200, q_bottom + 8), (lcx, q_bottom + 8), (lcx, py)], "arr-b")
    s += polyline([(720, q_bottom), (720, q_bottom + 8), (rcx, q_bottom + 8), (rcx, py)], "arr-b")

    edge_x = lx + pw
    i_out_y = y3 + 28
    pre_in_y = r1 + 29
    s += polyline(
        [(edge_x, i_out_y), (gutter, i_out_y), (gutter, pre_in_y), (rx + 14, pre_in_y)],
        "arr-g",
    )
    s += math_label_bg(
        gutter + 10,
        (i_out_y + pre_in_y) // 2 - 14,
        88,
        28,
        ["<tspan>index </tspan>", t_var("I", "", italic=True)],
    )

    by = 604
    s += box(80, by, 760, 92, "#f8fafc", "#64748b")
    s += plain(460, by + 20, "ESS: Latent-Cache 分层存储", cls="lb", anchor="middle")
    s += math_txt(
        260,
        by + 42,
        ["<tspan>GPU Sparse Memory Pool (LRU 热 </tspan>", t_c("KV"), "<tspan>)</tspan>"],
    )
    s += plain(660, by + 42, "CPU 冷 latent 池 (DRAM)", cls="dt", anchor="middle")
    s += plain(460, by + 60, "Indexer-Cache 始终在 GPU  |  仅 Latent-Cache 走 offload", cls="dt", anchor="middle")
    s += plain(460, by + 76, "Layer-wise DA/DBA overlap 掩盖 prefetch 延迟", cls="dt", anchor="middle")
    s += line(rcx, r4 + 64, 460, by, "arr-g")

    s += math_txt(
        460,
        758,
        [
            "<tspan>当前 </tspan>",
            t_var("q", "t"),
            "<tspan> 选历史 top-</tspan>",
            t_var("k", "", italic=False),
            "<tspan> 下标 </tspan>",
            t_var("I", "", italic=True),
            "<tspan> + Latent-Cache 仅对 </tspan>",
            t_var("I", "", italic=True),
            "<tspan> 内 token 做 MLA</tspan>",
        ],
        cls="an",
    )

    write("ess-dual-cache.svg", s, w, h, "ec", math=True)


def blt(x, y, s):
    return f'<text x="{x}" y="{y}" class="bl">{esc(s)}</text>'


def gen_lightning_indexer_forward() -> None:
    """Decode 一步 Indexer 前向: 固定 t 遍历 s, 输入输出 (tspan math)."""
    w, h = 920, 660
    s = ""
    s += math_txt(
        460,
        22,
        [
            t_cn("Lightning Indexer: Decode 一步前向 ("),
            t_var("t"),
            t_cn(" 固定, 遍历 "),
            t_var("s"),
            t_cn(")"),
        ],
        cls="fm t",
    )
    s += math_txt(
        460,
        40,
        [
            t_cn("例 "),
            t_loss(),
            t_cn("=8 已生成 | 当前 "),
            t_var("t", "9"),
            t_cn(" | 实际 "),
            t_loss(),
            t_cn(" 可达 128K, "),
            t_var("k", "", italic=False),
            t_cn("=2048"),
        ],
        cls="fm st",
        size="10px",
    )

    py, ph, pw = 58, 430, 272
    lx, mx, ox = 24, 312, 624
    lcx, mcx, ocx = lx + pw // 2, mx + pw // 2, ox + pw // 2

    # --- 输入 ---
    s += panel(lx, py, pw, ph, "", "#eff6ff", "#93c5fd")
    s += math_txt(
        lcx,
        py + 16,
        [t_cn("输入 (decode step "), t_var("t", "9"), t_cn(")")],
        cls="fm lb",
    )
    y = py + 36
    s += box(lx + 12, y, pw - 24, 44, "#dbeafe", "#2563eb")
    s += math_txt(lcx, y + 18, [t_var("h", "9"), t_cn("  [5120]")], cls="fm lb")
    s += plain(lcx, y + 34, "当前 token 隐状态 (本步唯一新算)", cls="dt", anchor="middle")
    y += 54
    s += math_txt(
        lcx,
        y,
        [t_cn("Indexer-Cache (GPU, 全长 "), t_loss(), t_cn(" 行)")],
        cls="fm lb",
    )
    y += 16
    s += box(lx + 12, y, pw - 24, 168, "#f8fafc", "#cbd5e1")
    rows = [(1, "1"), (2, "2"), (3, "3"), ("...", "..."), (8, "8")]
    ry = y + 18
    for rs, rk in rows:
        s += blt_math(
            lx + 24,
            ry,
            [
                t_srow(str(rs)),
                t_var("k", rk),
                t_cn("  ["),
                t_var("d", "I"),
                t_cn("]  (indexer key, 已缓存)"),
            ],
        )
        ry += 22
    s += math_txt(
        lcx,
        y + 152,
        [t_cn("每行 = 历史 token "), t_var("s"), t_cn(" 的 indexer "), t_var("K", "", italic=False)],
        cls="fm dt",
        size="9px",
    )
    y += 182
    s += box(lx + 12, y, pw - 24, 72, "#fff7ed", "#fdba74")
    s += math_txt(
        lcx,
        y + 18,
        [t_var("K", "", italic=False), t_cn(" 从哪来? (历史 write)")],
        cls="fm lb",
    )
    s += blt_math(
        lx + 22,
        y + 36,
        [t_cn("- token "), t_var("s"), t_cn(" 当时算完 "), t_var("h", "s")],
    )
    s += blt_math(
        lx + 22,
        y + 50,
        [
            t_cn("- "),
            t_var("k", "s"),
            t_cn(" = "),
            t_w("IK"),
            t_cn(" proj("),
            t_var("h", "s"),
            t_cn(")  append 第 "),
            t_var("s"),
            t_cn(" 行"),
        ],
    )
    s += blt_math(
        lx + 22,
        y + 64,
        [
            t_cn("- 本步 "),
            t_var("t", "9"),
            t_cn(" 只读, 不重算 "),
            t_var("k", "1"),
            t_cn(".."),
            t_var("k", "8"),
        ],
    )

    # --- 前向 ---
    s += panel(mx, py, pw, ph, "", "#fff7ed", "#fdba74")
    s += math_txt(
        mcx,
        py + 16,
        [
            t_cn("Indexer 前向 (固定 "),
            t_var("t"),
            t_cn(", 遍历 "),
            t_var("s"),
            t_cn(")"),
        ],
        cls="fm lb",
    )
    y = py + 36
    s += box(mx + 12, y, pw - 24, 52, "#ffedd5", "#f97316")
    s += math_txt(
        mcx,
        y + 18,
        [t_cn("Step A: 算当前 "), t_var("Q", "", italic=False)],
        cls="fm lb",
    )
    s += math_txt(
        mcx,
        y + 34,
        [
            t_var("q", "9"),
            t_cn(" = "),
            t_w("IQ"),
            t_cn(" slice("),
            t_var("h", "9"),
            t_cn(")   ["),
            t_var("H", "I"),
            t_cn(" heads "),
            t_times(),
            t_cn(" "),
            t_var("d", "I"),
            t_cn("]"),
        ],
        cls="fm",
        size="9px",
    )
    s += math_txt(
        mcx,
        y + 48,
        [t_cn("仅 query 侧现算; "), t_var("K", "", italic=False), t_cn(" 全在 cache")],
        cls="fm dt",
        size="9px",
    )
    y += 62
    s += box(mx + 12, y, pw - 24, 118, "#fff8ee", "#E67E22")
    s += math_txt(
        mcx,
        y + 18,
        [t_cn("Step B: 对每个历史 "), t_var("s"), t_cn(" 打分")],
        cls="fm lb",
    )
    s += blt_math(
        mx + 22,
        y + 36,
        [t_cn("for "), t_var("s"), t_cn(" = 1, 2, ..., "), t_loss(), t_cn(":")],
    )
    s += blt_math(
        mx + 34,
        y + 52,
        [
            t_idx("9,s"),
            t_cn(" = "),
            t_sum("j"),
            t_cn(" "),
            t_var("w", "9,j"),
            t_cn(" "),
            t_relu("".join([t_var("q", "9,j"), t_dot(), t_var("k", "s")])),
        ],
    )
    s += blt_math(
        mx + 34,
        y + 68,
        [t_cn("固定 "), t_var("t", "9"), t_cn(", "), t_var("s"), t_cn(" 从 1 扫到 "), t_loss()],
    )
    s += math_txt(
        mcx,
        y + 88,
        [t_loss(), t_cn("=8 次点积 "), t_arrow(), t_cn(" 8 个标量")],
        cls="fm dt",
        size="9px",
    )
    s += math_txt(
        mcx,
        y + 102,
        [
            t_var("q", "9"),
            t_cn(" 选 "),
            t_var("k", "s"),
            t_cn(", 不是 "),
            t_var("k", "s"),
            t_cn(" 选 "),
            t_var("q", "9"),
        ],
        cls="fm an",
        size="9px",
    )
    y += 128
    s += box(mx + 12, y, pw - 24, 44, "#f0fdf4", "#22c55e")
    s += math_txt(
        mcx,
        y + 18,
        [t_cn("Step C: Top-"), t_var("k", "", italic=False)],
        cls="fm lb",
    )
    s += math_txt(
        mcx,
        y + 34,
        [
            t_cn("取 "),
            t_idx("9,s"),
            t_cn(" 最高的 "),
            t_var("k", "", italic=False),
            t_cn("=2048 个下标 "),
            t_var("s"),
        ],
        cls="fm",
        size="9px",
    )
    y += 54
    s += box(mx + 12, y, pw - 24, 44, "#eef4fc", "#4A90D9")
    s += plain(mcx, y + 18, "DeepGEMM / paged kernel", cls="lb", anchor="middle")
    s += math_txt(
        mcx,
        y + 34,
        [
            t_cn("batch dot("),
            t_var("q", "9"),
            t_cn(", Indexer-Cache) "),
            t_arrow(),
            t_cn(" ["),
            t_loss(),
            t_cn("]"),
        ],
        cls="fm",
        size="9px",
    )
    y += 54
    s += box(mx + 12, y, pw - 24, 56, "#fff0f0", "#e8a0a0")
    s += plain(mcx, y + 18, "Indexer 不算", cls="lb", anchor="middle")
    s += math_txt(
        mcx,
        y + 34,
        [t_cn("softmax / "), t_var("V"), t_cn(" / 512-d "), t_c("KV")],
        cls="fm",
        size="9px",
    )
    s += plain(mcx, y + 48, "Latent-Cache 升维在 Core MLA", cls="dt", anchor="middle")

    # --- 输出 ---
    s += panel(ox, py, pw, ph, "", "#f0fdf4", "#86efac")
    s += plain(ocx, py + 16, "输出", cls="lb", anchor="middle")
    y = py + 36
    s += box(ox + 12, y, pw - 24, 64, "#dcfce7", "#16a34a")
    s += math_txt(
        ocx,
        y + 18,
        [t_cn("分数向量 (长度 "), t_loss(), t_cn(")")],
        cls="fm lb",
    )
    s += math_txt(
        ocx,
        y + 34,
        [
            t_cn("["),
            t_idx("9,1"),
            t_cn(", "),
            t_idx("9,2"),
            t_cn(", ..., "),
            t_idx("9,8"),
            t_cn("]"),
        ],
        cls="fm",
        size="9px",
    )
    s += math_txt(
        ocx,
        y + 50,
        [
            t_cn("shape ["),
            t_loss(),
            t_cn("], 每个 "),
            t_var("s"),
            t_cn(" 一个 "),
            t_idx("t,s"),
        ],
        cls="fm",
        size="9px",
    )
    y += 74
    s += box(ox + 12, y, pw - 24, 64, "#bbf7d0", "#22c55e")
    s += math_txt(
        ocx,
        y + 18,
        [t_cn("index 集 "), t_var("I", "", italic=True)],
        cls="fm lb",
    )
    s += math_txt(
        ocx,
        y + 34,
        [
            t_var("I", "", italic=True),
            t_cn(" = {3, 5, 8, ...}  ("),
            t_var("k", "", italic=False),
            t_cn(" 个历史下标 "),
            t_var("s"),
            t_cn(")"),
        ],
        cls="fm",
        size="9px",
    )
    s += math_txt(
        ocx,
        y + 50,
        [
            t_cn("例 "),
            t_loss(),
            t_cn("=8 时 "),
            t_var("k", "", italic=False),
            t_cn("=min(2048,"),
            t_loss(),
            t_cn(")=8"),
        ],
        cls="fm",
        size="9px",
    )
    y += 74
    s += box(ox + 12, y, pw - 24, 72, "#eff6ff", "#93c5fd")
    s += plain(ocx, y + 18, "交给 Core MLA", cls="lb", anchor="middle")
    s += math_txt(
        ocx,
        y + 34,
        [
            t_cn("只读 Latent-Cache 中 "),
            t_var("s"),
            t_in(),
            t_var("I", "", italic=True),
            t_cn(" 的行"),
        ],
        cls="fm",
        size="9px",
    )
    s += math_txt(
        ocx,
        y + 48,
        [t_cn("升维 "), t_var("K", "", italic=False), t_cn("/"), t_var("V"), t_cn(" + softmax attention")],
        cls="fm",
        size="9px",
    )
    s += math_txt(
        ocx,
        y + 62,
        [t_cn("输出 "), t_var("u", "9"), t_arrow(), t_cn(" 下一层 / 写 "), t_c("KV", "9")],
        cls="fm",
        size="9px",
    )
    y += 86
    s += box(ox + 12, y, pw - 24, 56, "#f8fafc", "#64748b")
    s += plain(ocx, y + 18, "本步同时写入 (非 Indexer 输出)", cls="lb", anchor="middle")
    s += math_txt(
        ocx,
        y + 34,
        [t_var("k", "9"), t_cn(" "), t_arrow(), t_cn(" Indexer-Cache 第 9 行")],
        cls="fm",
        size="9px",
    )
    s += math_txt(
        ocx,
        y + 48,
        [t_c("KV", "9"), t_arrow(), t_cn(" Latent-Cache 第 9 行")],
        cls="fm",
        size="9px",
    )

    # --- 箭头 ---
    s += line(lx + pw, py + 70, mx, py + 70, "arr-b")
    s += line(lx + pw, py + 160, mx, py + 160, "arr-b")
    arrow_y = py + 120
    s += line(mx + pw, arrow_y, ox, arrow_y, "arr-g")
    gap_cx = (mx + pw + ox) // 2
    label_w = 48
    s += math_label_bg(gap_cx - label_w // 2, arrow_y - 11, label_w, 22, [t_idx("t,s")])

    # --- 底栏 ---
    by = 508
    s += box(40, by, 840, 88, "#f8fafc", "#94a3b8")
    s += math_txt(
        460,
        by + 18,
        [t_var("t"), t_cn(" 与 "), t_var("s"), t_cn(" 分工 (一层 decode 步)")],
        cls="fm lb",
    )
    s += math_txt(
        230,
        by + 40,
        [
            t_var("t"),
            t_cn(" = 当前步 (1 个 "),
            t_var("h", "t"),
            t_cn(", 1 个 "),
            t_var("q", "t"),
            t_cn(")"),
        ],
        cls="fm",
        size="9px",
    )
    s += math_txt(
        690,
        by + 40,
        [
            t_var("s"),
            t_cn(" = 历史槽位 ("),
            t_loss(),
            t_cn(" 个 "),
            t_var("k", "s"),
            t_cn(" 已在 cache)"),
        ],
        cls="fm",
        size="9px",
    )
    s += math_txt(
        460,
        by + 58,
        [
            t_cn("操作: 固定 "),
            t_var("t"),
            t_cn(", for "),
            t_var("s"),
            t_cn(" in 1.."),
            t_loss(),
            t_cn(" 得 "),
            t_idx("t,s"),
            t_cn("; Top-k 得 "),
            t_var("I", "", italic=True),
            t_cn("; Core MLA 用 "),
            t_var("I", "", italic=True),
        ],
        cls="fm",
        size="9px",
    )
    s += math_txt(
        460,
        by + 74,
        [
            t_cn("Prefill 同理: 当前行 "),
            t_var("t"),
            t_cn(" 对 "),
            t_var("s"),
            t_cn("<"),
            t_var("t"),
            t_cn(" 打分; 每行结束 append "),
            t_var("k", "t"),
        ],
        cls="fm an",
        size="9px",
    )

    s += plain(460, 638, "完整 MLA+Latent 分工见 ess-dual-cache.svg", cls="an", anchor="middle")

    write("lightning-indexer-forward.svg", s, w, h, "lif", math=True)


def gen_lightning_indexer_t_s_direction() -> None:
    """§3 位置 t/s、Top-k 方向示意（原 ASCII 图）。"""
    w, h = 920, 340
    s = ""
    s += math_txt(
        460,
        22,
        [
            t_cn("位置 "),
            t_var("t"),
            t_cn(" / "),
            t_var("s"),
            t_cn(" 与 Top-"),
            t_var("k", "", italic=False),
            t_cn(" 方向"),
        ],
        cls="fm t",
    )
    s += math_txt(
        460,
        40,
        [
            t_cn("固定当前 "),
            t_var("q", "t"),
            t_cn("，对每条历史 "),
            t_var("k", "s"),
            t_cn(" 打分 -> Top-"),
            t_var("k", "", italic=False),
        ],
        cls="fm st",
        size="10px",
    )

    # 历史 cache 行
    y_row = 88
    slots = [(120, "1"), (220, "2"), (320, "3"), (420, "..."), (520, "L")]
    for cx, lab in slots:
        s += box(cx - 36, y_row, 72, 52, "#f0f4ff", "#a0b8e8")
        if lab != "...":
            s += math_txt(cx, y_row + 38, [t_var("k", lab)], cls="fm dt", size="9px")
        else:
            s += txt(cx, y_row + 38, "...", "dt")
        if lab != "...":
            s += math_txt(cx, y_row + 20, [t_cn("s="), t_var(lab, "", italic=False)], cls="fm lb")
        else:
            s += txt(cx, y_row + 24, "...", "lb")

    s += txt(620, y_row + 18, "Indexer-Cache", "an")
    s += txt(620, y_row + 34, "历史行", "dt")

    # 当前 t
    s += box(760, y_row, 100, 52, "#eef4fc", "#4A90D9")
    s += math_txt(810, y_row + 20, [t_cn("当前 "), t_var("t")], cls="fm lb")
    s += math_txt(810, y_row + 38, [t_var("q", "t")], cls="fm dt", size="9px")

    # 打分方向箭头 q_t -> 各 k_s
    s += line(712, y_row + 26, 556, y_row + 26, "arr-b")
    s += math_txt(680, y_row + 14, [t_var("q", "t"), t_cn(" 打分")], cls="fm an", size="9px")

    # Top-k
    y_topk = 178
    s += line(320, y_row + 52, 320, y_topk - 28, "arr")
    s += line(520, y_row + 52, 520, y_topk - 28, "arr")
    s += node(460, y_topk, 200, 48, "Top-k Selector", ["取分数最高的 k 个历史下标 s"], "#f0faf0", "#27AE60")

    y_out = 248
    s += line(460, y_topk + 24, 460, y_out - 28, "arr-g")
    s += math_txt(
        460,
        y_out,
        [t_cn("index 集 "), t_var("I", "", italic=False), t_cn("（"), t_var("k", "", italic=False), t_cn(" 个 "), t_var("s", "", italic=False), t_cn("）")],
        cls="fm lb",
    )
    s += math_txt(
        460,
        y_out + 18,
        [t_cn("Core MLA 只 attend 这 "), t_var("k", "", italic=False), t_cn(" 个历史位置")],
        cls="fm dt",
        size="9px",
    )

    # 多头 j（与 s 无关）
    y_head = 300
    s += box(40, y_head - 20, 840, 36, "#fffbeb", "#f59e0b", rx=4)
    s += math_txt(
        460,
        y_head + 2,
        [
            t_cn("多头 indexer："),
            t_supsub("q", "t,1"),
            t_cn(", …, "),
            t_supsub("q", "t,j"),
            t_cn(", … — "),
            t_var("j", "", italic=False),
            t_cn(" 是头编号，"),
            t_cn("≠ 历史下标 "),
            t_var("s"),
        ],
        cls="fm dt",
        size="10px",
    )

    write("lightning-indexer-t-s-direction.svg", s, w, h, "litsd", math=True)


def gen_dsa_three_stage() -> None:
    """每层三阶段竖向数据流（对应 dsa-sparse-attention.md 表下 ASCII 图）。"""
    w, h = 920, 320
    cx = 460
    s = ""
    s += txt(cx, 22, "DSA 每层三阶段数据流", "t")
    s += txt(cx, 40, "Query + 全长历史 -> Indexer -> Top-k -> Core MLA", "st")

    y_idx = 78
    s += node(200, y_idx, 128, 56, "Query h_t", ["当前 token"], "#eef4fc", "#4A90D9")
    s += node(cx, y_idx, 168, 56, "Lightning Indexer", ["对全长历史打分"], "#fff8ee", "#E67E22")
    s += node(720, y_idx, 148, 56, "全长 latent 历史", ["Indexer-Cache"], "#f0f4ff", "#a0b8e8")
    s += line(264, y_idx, 376, y_idx)
    s += line(544, y_idx, 646, y_idx)

    y_topk = 168
    s += line(cx, y_idx + 28, cx, y_topk - 26, "arr")
    s += txt(cx + 72, 148, "top-k (2048)", "an")
    s += node(cx, y_topk, 168, 52, "Top-k Selector", ["index 集 I"], "#f0faf0", "#27AE60")

    y_core = 228
    s += line(cx, y_topk + 26, cx, y_core - 28, "arr")
    s += node(cx, y_core, 180, 56, "Core MLA Attention", ["仅 I 内 latent"], "#eef4fc", "#4A90D9")
    s += line(cx + 90, y_core, 800, y_core, "arr-g")
    s += txt(850, y_core + 4, "输出", "lb")

    y_lat = 298
    s += node(cx, y_lat, 200, 48, "Latent-Cache", ["被选中的 entries"], "#f0f4ff", "#a0b8e8")
    s += line(cx, y_lat - 24, cx, y_core + 28, "arr-b")

    write("dsa-three-stage.svg", s, w, h, "dts")


def main() -> None:
    gen_dsa_pipeline()
    gen_dsa_three_stage()
    gen_index_share_fffs()
    gen_ess_dual_cache()
    gen_lightning_indexer_forward()
    gen_lightning_indexer_t_s_direction()


if __name__ == "__main__":
    main()
