#!/usr/bin/env python3
"""Generate engram-01d-multi-head-hash.svg — canonical math-in-SVG example.

Math helpers: reference/svg_math_helpers.py
Full spec: reference/README.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
from _paths import ENGRAM_DIAGRAMS  # noqa: E402
import xml.etree.ElementTree as ET
from svg_math_helpers import (  # noqa: E402
    LABEL_FONT,
    MATH_FONT,
    PHI,
    default_math_styles,
    math_line,
    plain,
    sub,
    t_greek,
    t_var,
)

OUT = ENGRAM_DIAGRAMS / "engram-01d-multi-head-hash.svg"
W, H = 720, 336
DIV_Y = 228  # horizontal CPU | GPU boundary


def main() -> None:
    formula = math_line(
        W // 2,
        38,
        [
            t_var("z", "t,n,k"),
            "<tspan> = </tspan>",
            t_greek("phi", "n,k"),
            "<tspan>(</tspan>",
            t_var("g", "t,n"),
            "<tspan>),  </tspan>",
            t_var("e", "t,n,k"),
            "<tspan> = </tspan>",
            t_var("E", "n,k"),
            "<tspan>[</tspan>",
            t_var("z", "t,n,k"),
            "<tspan>]  |  deterministic multiply-XOR hash</tspan>",
        ],
    )

    s = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="{MATH_FONT}">
  <defs>
    <marker id="mh_a" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#475569"/>
    </marker>
    <style>
{default_math_styles()}
      .arr{{stroke:#475569;stroke-width:1.2;fill:none;marker-end:url(#mh_a)}}
      .tbl{{fill:#faf5ff;stroke:#a78bfa;stroke-width:1}}
      .cell{{fill:#ede9fe;stroke:#c4b5fd;stroke-width:0.6}}
      .hit{{fill:#7c3aed;stroke:#5b21b6;stroke-width:0.8}}
      .note{{fill:#f0fdf4;stroke:#86efac;stroke-width:1.2;rx:6}}
      .zone-cpu{{fill:#eff6ff;stroke:#93c5fd;stroke-width:1;opacity:0.55}}
      .zone-gpu{{fill:#fff7ed;stroke:#fb923c;stroke-width:1;opacity:0.55}}
      .divln{{stroke:#64748b;stroke-width:1.4;stroke-dasharray:6 4}}
      .zh{{font-size:9px;font-weight:700;fill:#1e40af;font-family:{LABEL_FONT}}}
      .zg{{font-size:9px;font-weight:700;fill:#c2410c;font-family:{LABEL_FONT}}}
      .pref{{stroke:#ea580c;stroke-width:1.2;fill:none;stroke-dasharray:4 3;marker-end:url(#mh_a)}}
    </style>
  </defs>

  <!-- CPU / GPU zones -->
  <rect x="12" y="48" width="{W - 24}" height="{DIV_Y - 52}" rx="8" class="zone-cpu"/>
  <rect x="12" y="{DIV_Y + 6}" width="{W - 24}" height="{H - DIV_Y - 22}" rx="8" class="zone-gpu"/>
  <line x1="12" y1="{DIV_Y}" x2="{W - 12}" y2="{DIV_Y}" class="divln"/>
  <text x="24" y="{DIV_Y - 6}" class="zh" text-anchor="start">CPU / Host DRAM / CXL</text>
  <text x="24" y="{DIV_Y + 16}" class="zg" text-anchor="start">GPU HBM (prefetch staging)</text>

  <text x="360" y="20" class="t">Step 3: Multi-head Hash Addressing</text>
{formula}
  <rect x="16" y="72" width="96" height="44" class="box" fill="#f1f5f9" stroke="#94a3b8"/>
'''
    s += sub(64, 92, "g", "t,n", cls="m")
    s += plain(64, 106, "n-gram at t", cls="s")
    s += plain(64, 118, "e.g. 3-gram", cls="s")

    s += '''  <line x1="112" y1="94" x2="132" y2="94" class="arr"/>
  <line x1="132" y1="94" x2="132" y2="58" stroke="#94a3b8" stroke-width="1.2" fill="none"/>
  <line x1="132" y1="94" x2="132" y2="130" stroke="#94a3b8" stroke-width="1.2" fill="none"/>
  <line x1="132" y1="94" x2="132" y2="202" stroke="#94a3b8" stroke-width="1.2" fill="none"/>
  <line x1="132" y1="58" x2="148" y2="58" class="arr"/>
  <line x1="132" y1="130" x2="148" y2="130" class="arr"/>
  <line x1="132" y1="202" x2="148" y2="202" class="arr"/>
  <text x="124" y="88" class="fm" font-style="italic">K</text>
  <text x="134" y="88" class="fm"> heads</text>
'''

    rows = [
        (58, "1", 334, 46),
        (130, "2", 366, 118),
        (202, "K", 318, 190),
    ]
    for cy, k, hit_col, _y_tbl in rows:
        s += f'  <rect x="148" y="{cy-16}" width="72" height="32" class="box" fill="#fef3c7" stroke="#d97706"/>\n'
        s += (
            f'  <text x="184" y="{cy}" class="m" text-anchor="middle">'
            f'<tspan font-style="italic">{PHI}</tspan>'
            f'<tspan baseline-shift="sub" font-size="0.72em">n,{k}</tspan>'
            f"</text>\n"
        )
        s += plain(184, cy + 12, f"hash head {k}", cls="s")
        s += f'  <line x1="220" y1="{cy}" x2="236" y2="{cy}" class="arr"/>\n'
        s += f'  <rect x="236" y="{cy-12}" width="52" height="24" class="box" fill="#fff7ed" stroke="#fb923c"/>\n'
        s += (
            f'  <text x="262" y="{cy+4}" class="m" text-anchor="middle">'
            f'<tspan font-style="italic">z</tspan>'
            f'<tspan baseline-shift="sub" font-size="0.72em">t,n,{k}</tspan>'
            f"</text>\n"
        )
        s += f'  <line x1="288" y1="{cy}" x2="304" y2="{cy}" class="arr"/>\n'
        s += f'  <rect x="304" y="{cy-18}" width="108" height="36" class="tbl"/>\n'
        for i in range(6):
            s += f'  <rect x="{310+i*14}" y="{cy-12}" width="14" height="10" class="cell"/>\n'
            s += f'  <rect x="{310+i*14}" y="{cy-2}" width="14" height="10" class="cell"/>\n'
        s += f'  <rect x="{hit_col}" y="{cy-2}" width="14" height="10" class="hit"/>\n'
        s += (
            f'  <text x="358" y="{cy-14}" class="fm" text-anchor="middle">'
            f'<tspan font-style="italic">E</tspan>'
            f'<tspan baseline-shift="sub" font-size="0.68em">n,{k}</tspan>'
            f"</text>\n"
        )
        s += f'  <line x1="412" y1="{cy}" x2="428" y2="{cy}" class="arr"/>\n'
        s += f'  <rect x="428" y="{cy-12}" width="56" height="24" class="box" fill="#dcfce7" stroke="#16a34a"/>\n'
        s += (
            f'  <text x="456" y="{cy+4}" class="m" text-anchor="middle">'
            f'<tspan font-style="italic">e</tspan>'
            f'<tspan baseline-shift="sub" font-size="0.72em">t,n,{k}</tspan>'
            f"</text>\n"
        )
        s += f'  <line x1="484" y1="{cy}" x2="512" y2="{cy}" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="3 2"/>\n'

    s += f'''  <line x1="512" y1="58" x2="512" y2="202" stroke="#cbd5e1" stroke-width="1.2"/>
  <line x1="512" y1="130" x2="528" y2="130" class="arr"/>
  <rect x="528" y="108" width="72" height="44" class="box" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="564" y="126" class="b" text-anchor="middle" font-family="{LABEL_FONT}">concat</text>
  <text x="564" y="142" class="s" text-anchor="middle">Step 4: <tspan font-style="italic">e</tspan><tspan baseline-shift="sub" font-size="0.72em">t</tspan></text>

  <!-- CPU: pre-calculable (right) -->
  <rect x="468" y="168" width="108" height="52" class="note"/>
  <text x="522" y="188" class="b" fill="#166534" text-anchor="middle" font-family="{LABEL_FONT}">Pre-calculable</text>
  <text x="522" y="202" class="s" font-size="8" fill="#15803d" text-anchor="middle">index = f(input ids)</text>
  <text x="522" y="214" class="s" font-size="8" fill="#15803d" text-anchor="middle">lookup on CPU / CXL</text>

  <!-- prefetch across CPU|GPU boundary -->
  <path d="M 564 152 L 564 {DIV_Y - 2} L 564 {DIV_Y + 28}" class="pref"/>
  <text x="576" y="{(DIV_Y + DIV_Y + 28) // 2 + 4}" class="s" font-size="7" fill="#c2410c" text-anchor="start">prefetch</text>

  <rect x="514" y="{DIV_Y + 36}" width="100" height="40" class="box" fill="#ffedd5" stroke="#ea580c" stroke-width="1.3"/>
  <text x="564" y="{DIV_Y + 54}" class="b" text-anchor="middle" font-family="{LABEL_FONT}">
    <tspan font-style="italic">e</tspan><tspan baseline-shift="sub" font-size="0.72em">t</tspan> staging
  </text>
  <text x="564" y="{DIV_Y + 68}" class="s" font-size="8" text-anchor="middle">VRAM buffer</text>

  <text x="200" y="{DIV_Y + 54}" class="s" font-size="8" fill="#c2410c" text-anchor="start">Step 5+ on GPU: W_K, W_V, gate alpha (needs h_t)</text>

  <rect x="16" y="168" width="96" height="52" rx="6" fill="#dbeafe" stroke="#3b82f6" stroke-width="1"/>
  <text x="64" y="186" class="s" font-size="8" text-anchor="middle">prime-sized</text>
  <text x="64" y="200" class="s" font-size="8" text-anchor="middle">tables per head</text>
  <text x="64" y="214" class="s" font-size="8" text-anchor="middle">Host DRAM / CXL</text>

  <text x="360" y="{H - 10}" class="s" text-anchor="middle">each order <tspan font-style="italic">n</tspan> has <tspan font-style="italic">K</tspan> independent heads; full n-gram space too large for explicit tabulation</text>
</svg>
'''
    OUT.write_text(s, encoding="utf-8")
    errs: list[str] = []
    try:
        root = ET.parse(OUT).getroot()
        if not root.attrib.get("viewBox"):
            errs.append("missing viewBox")
    except ET.ParseError as e:
        errs.append(str(e))
    if errs:
        raise SystemExit(errs)
    print("OK", OUT)


if __name__ == "__main__":
    main()
