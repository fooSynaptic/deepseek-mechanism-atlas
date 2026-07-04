#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ternary plot: 模型 / 架构-train / 架构-infer optimization axes."""
from __future__ import annotations

import math
from collections import defaultdict

from gen_deepseek_svgs import esc, svg_header, write  # noqa: E402

# (model %, arch-train %, arch-infer %, short label)
WORKS: list[tuple[int, int, int, str]] = [
    (100, 0, 0, "V1"),
    (100, 0, 0, "R1"),
    (0, 40, 60, "MLA"),
    (0, 45, 55, "MoE"),
    (0, 50, 50, "V3 路由"),
    (0, 70, 30, "MTP"),
    (0, 100, 0, "FP8"),
    (80, 0, 20, "V3.1"),
    (0, 0, 100, "Terminus"),
    (0, 35, 65, "DSA"),
    (0, 0, 100, "Index Share"),
    (0, 0, 100, "ESS"),
    (0, 30, 70, "CSA/HCA"),
    (0, 55, 45, "mHC"),
    (0, 100, 0, "Muon"),
    (0, 55, 45, "Hash MoE"),
    (0, 0, 100, "DSpark"),
    (0, 0, 100, "HiSparse"),
    (0, 0, 100, "FlashMLA"),
    (60, 25, 15, "Visual"),
]

# Triangle vertices: Model (top), Arch-train (BL), Arch-infer (BR)
AX, AY = 490, 118
BX, BY = 120, 448
CX, CY = 860, 448

CENT_X = (AX + BX + CX) / 3
CENT_Y = (AY + BY + CY) / 3


def bary_xy(m: int, t: int, i: int) -> tuple[float, float]:
    s = m + t + i
    if s <= 0:
        return CENT_X, CENT_Y
    mf, tf, inf = m / s, t / s, i / s
    x = mf * AX + tf * BX + inf * CX
    y = mf * AY + tf * BY + inf * CY
    return x, y


def spread_point(key: tuple[int, int, int], idx: int, n: int) -> tuple[float, float]:
    x, y = bary_xy(*key)
    if n <= 1:
        return x, y
    base_r = 26 if max(key) == 100 else 18
    r = base_r + max(0, n - 3) * 3
    ang = -math.pi / 2 + 2 * math.pi * idx / n
    return x + r * math.cos(ang), y + r * math.sin(ang)


def label_anchor(
    px: float,
    py: float,
    idx: int,
    n: int,
    label: str,
    key: tuple[int, int, int],
) -> tuple[float, float, str]:
    """Place label outward from triangle centroid; fan when clustered."""
    dx, dy = px - CENT_X, py - CENT_Y
    dist = math.hypot(dx, dy) or 1.0
    ux, uy = dx / dist, dy / dist
    px_perp, py_perp = -uy, ux

    char_w = 5.5 if any(ord(c) > 127 for c in label) else 3.2
    along = 36 + min(len(label) * char_w * 0.4, 28)
    spread = 16

    # Dense 100% infer cluster at bottom-right vertex
    near_c = math.hypot(px - CX, py - CY) < 42
    if key == (0, 0, 100) and near_c and n > 1:
        along = 44 + len(label) * 2.2
        spread = 20
        ang0 = -math.pi / 6
        ang1 = math.pi / 3
        ang = ang0 + (ang1 - ang0) * idx / max(n - 1, 1)
        lx = px + along * math.cos(ang)
        ly = py + along * math.sin(ang)
        anchor = "start" if math.cos(ang) >= 0 else "end"
        return lx, ly, anchor

    # 100% model cluster at top vertex
    near_a = math.hypot(px - AX, py - AY) < 36
    if key == (100, 0, 0) and near_a and n > 1:
        along = 40
        spread = 22
        off = (idx - (n - 1) / 2) * spread
        lx = px + off
        ly = py - along
        return lx, ly, "middle"

    if n > 1:
        along += 12
        off = (idx - (n - 1) / 2) * spread
    else:
        off = 0.0

    lx = px + ux * along + px_perp * off
    ly = py + uy * along + py_perp * off

    if abs(ux) >= abs(uy):
        anchor = "start" if ux > 0 else "end"
    else:
        anchor = "middle"
    if n > 1 and abs(off) > 8:
        anchor = "start" if off > 0 else "end" if off < 0 else anchor

    return lx, ly, anchor


def point_color(m: int, t: int, i: int) -> str:
    if m >= 50:
        return "#2563eb"
    if t >= 50 and t >= i:
        return "#d97706"
    if i >= 50:
        return "#7c3aed"
    return "#475569"


def grid_const(comp: str, f: float, steps: int = 24) -> str:
    pts: list[tuple[float, float]] = []
    for k in range(steps + 1):
        u = k / steps
        rest = 1.0 - f
        if comp == "a":
            a, b, c = f, u * rest, (1.0 - u) * rest
        elif comp == "b":
            b = f
            a, c = u * rest, (1.0 - u) * rest
        else:
            c = f
            a, b = u * rest, (1.0 - u) * rest
        x = a * AX + b * BX + c * CX
        y = a * AY + b * BY + c * CY
        pts.append((x, y))
    d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return f'<path d="{d}" fill="none" stroke="#d8dee9" stroke-width="1" stroke-dasharray="4 5"/>'


def gen_opt_direction_ternary() -> None:
    w, h = 980, 580
    s = ""
    s += f'<text x="{w // 2}" y="30" class="t">优化方向三轴分布</text>'
    s += (
        f'<text x="{w // 2}" y="50" class="st">'
        "各工作按 模型 / 架构-train / 架构-infer 占比落点（占比见 1.1 表）"
        "</text>"
    )

    s += (
        f'<polygon points="{AX},{AY} {BX},{BY} {CX},{CY}" '
        'fill="#f6f8fc" stroke="#94a3b8" stroke-width="2"/>'
    )

    for frac in (0.5,):
        s += grid_const("a", frac)
        s += grid_const("b", frac)
        s += grid_const("c", frac)

    s += f'<text x="{AX}" y="{AY - 22}" class="lb" fill="#2563eb">模型</text>'
    s += f'<text x="{AX}" y="{AY - 6}" class="dt">权重/数据/后训练</text>'
    s += f'<text x="{BX - 10}" y="{BY + 24}" class="lb" fill="#d97706" text-anchor="end">架构-train</text>'
    s += f'<text x="{BX - 10}" y="{BY + 40}" class="dt" text-anchor="end">训推结构/重训算子</text>'
    s += f'<text x="{CX + 10}" y="{BY + 24}" class="lb" fill="#7c3aed" text-anchor="start">架构-infer</text>'
    s += f'<text x="{CX + 10}" y="{BY + 40}" class="dt" text-anchor="start">推理路径/infer 补丁</text>'

    groups: dict[tuple[int, int, int], list[tuple[int, int, int, str]]] = defaultdict(list)
    for item in WORKS:
        groups[(item[0], item[1], item[2])].append(item)

    for key in sorted(groups.keys(), key=lambda k: (-k[0], -k[1], -k[2])):
        items = groups[key]
        m, t, i = key
        col = point_color(m, t, i)
        n = len(items)
        for j, (_, _, _, label) in enumerate(items):
            px, py = spread_point(key, j, n)
            lx, ly, anchor = label_anchor(px, py, j, n, label, key)
            s += (
                f'<line x1="{px:.1f}" y1="{py:.1f}" x2="{lx:.1f}" y2="{ly:.1f}" '
                f'stroke="{col}" stroke-width="1" stroke-opacity="0.45"/>'
            )
            s += f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{col}" stroke="#fff" stroke-width="1.5"/>'
            s += (
                f'<text x="{lx:.1f}" y="{ly + 3:.1f}" class="an" fill="{col}" '
                f'text-anchor="{anchor}">{esc(label)}</text>'
            )

    ly0 = 518
    s += f'<line x1="48" y1="{ly0 - 12}" x2="{w - 48}" y2="{ly0 - 12}" stroke="#e2e8f0" stroke-width="1"/>'
    keys = [
        ("#2563eb", "模型轴主导 (>=50%)"),
        ("#d97706", "架构-train 主导"),
        ("#7c3aed", "架构-infer 主导"),
        ("#475569", "多轴混合 (无单一 >=50%)"),
    ]
    for k, (col, leg) in enumerate(keys):
        x = 56 + k * 228
        s += f'<circle cx="{x}" cy="{ly0 + 10}" r="5" fill="{col}"/>'
        s += f'<text x="{x + 14}" y="{ly0 + 14}" class="dt" text-anchor="start">{esc(leg)}</text>'
    s += (
        f'<text x="{w // 2}" y="{ly0 + 40}" class="st">'
        "同占比多点扇形展开；标签经引出线外置，100% 单轴工作聚于对应顶点"
        "</text>"
    )

    write("opt-direction-ternary.svg", s, w, h, prefix="opt")


if __name__ == "__main__":
    gen_opt_direction_ternary()
