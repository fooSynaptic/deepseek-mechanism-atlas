#!/usr/bin/env python3
"""Generate engram-00-series-lineage.svg — four-paper series evolution (no mermaid)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
from _paths import ENGRAM_DIAGRAMS  # noqa: E402

import xml.etree.ElementTree as ET
from pathlib import Path

OUT = ENGRAM_DIAGRAMS / "engram-00-series-lineage.svg"
W, H = 920, 268


def box(x: int, y: int, w: int, h: int, fill: str, stroke: str, sw: str = "1.2") -> str:
    return f'  <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" class="box" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>\n'


def label(x: int, y: int, line1: str, line2: str, line3: str, *, bold: bool = False) -> str:
    fw = ' font-weight="700"' if bold else ""
    return (
        f'  <text x="{x}" y="{y}" class="b"{fw} text-anchor="middle">{line1}</text>\n'
        f'  <text x="{x}" y="{y + 14}" class="s" text-anchor="middle">{line2}</text>\n'
        f'  <text x="{x}" y="{y + 26}" class="id" text-anchor="middle">{line3}</text>\n'
    )


def main() -> None:
    # columns: root | papers | layer notes
    x0, w0 = 24, 148
    x1, w1 = 220, 168
    x2, w2 = 500, 168
    bh = 52
    ys = [36, 108, 180]  # B, C, D rows
    y_mid = 108  # center A

    s = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="system-ui, -apple-system, sans-serif">
  <defs>
    <marker id="ln_a" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#475569"/>
    </marker>
    <style>
      .t{{font-size:14px;font-weight:700;fill:#0f172a;text-anchor:middle}}
      .b{{font-size:10px;fill:#1e293b;text-anchor:middle}}
      .s{{font-size:9px;fill:#475569;text-anchor:middle}}
      .id{{font-size:8px;fill:#64748b;text-anchor:middle;font-family:ui-monospace,monospace}}
      .tag{{font-size:8px;font-weight:600;fill:#7c3aed;text-anchor:middle}}
      .box{{rx:6}}
      .arr{{stroke:#64748b;stroke-width:1.2;fill:none;marker-end:url(#ln_a)}}
      .dash{{stroke:#94a3b8;stroke-width:1.1;fill:none;stroke-dasharray:4 3;marker-end:url(#ln_a)}}
    </style>
  </defs>

  <text x="{W // 2}" y="22" class="t">Engram 系列演进关系</text>

  <!-- root -->
'''
    s += box(x0, y_mid - 4, w0, bh + 8, "#ede9fe", "#7c3aed", "1.4")
    s += label(x0 + w0 // 2, y_mid + 10, "① Engram", "架构奠基", "2601.07372", bold=True)

    papers = [
        (ys[0], "② CXL Pooling", "多机内存池", "2603.10087", "#eff6ff", "#3b82f6"),
        (ys[1], "③ Engram-Nine", "热层无碰撞实验", "2601.16531", "#ecfdf5", "#16a34a"),
        (ys[2], "④ Tiny-Engram", "视觉模态", "2605.20309", "#fff7ed", "#ea580c"),
    ]
    layers = [
        (ys[0], "部署层", "SGLang + CXL Switch", "#f0fdf4", "#22c55e"),
        (ys[1], "训练层", "碰撞 = 隐式正则", "#f8fafc", "#94a3b8"),
        (ys[2], "应用层", "SD / DiT 概念注入", "#faf5ff", "#a78bfa"),
    ]

    for y, t1, t2, t3, fill, stroke in papers:
        cy = y + bh // 2
        s += f'  <line x1="{x0 + w0}" y1="{y_mid + bh // 2}" x2="{x1}" y2="{cy}" class="arr"/>\n'
        s += box(x1, y, w1, bh, fill, stroke)
        s += label(x1 + w1 // 2, y + 14, t1, t2, t3)

    for y, tag, note, fill, stroke in layers:
        cy = y + bh // 2
        s += f'  <line x1="{x1 + w1}" y1="{cy}" x2="{x2}" y2="{cy}" class="dash"/>\n'
        s += f'  <text x="{(x1 + w1 + x2) // 2}" y="{cy - 6}" class="tag">{tag}</text>\n'
        s += box(x2, y, w2, bh, fill, stroke)
        s += label(x2 + w2 // 2, y + 18, note, "", "")

    s += f'''  <text x="{W // 2}" y="{H - 10}" class="s">奠基论文分出系统 / 训练 / 视觉三条线；虚线 = 各篇侧重层级</text>
</svg>
'''
    OUT.write_text(s, encoding="utf-8")
    root = ET.parse(OUT).getroot()
    if not root.attrib.get("viewBox"):
        raise SystemExit(["missing viewBox"])
    print("OK", OUT)


if __name__ == "__main__":
    main()
