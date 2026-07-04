#!/usr/bin/env python3
"""Generate engram-01e-residual-step78.svg: Step 7 ShortConv residual + Step 8 writeback."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
from _paths import ENGRAM_DIAGRAMS  # noqa: E402

import xml.etree.ElementTree as ET
from pathlib import Path

OUT = ENGRAM_DIAGRAMS / "engram-01e-residual-step78.svg"
W, H = 860, 340


def main() -> None:
    s = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="system-ui, -apple-system, sans-serif">
  <defs>
    <marker id="rs_a" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#475569"/>
    </marker>
    <style>
      .t{{font-size:14px;font-weight:700;fill:#0f172a;text-anchor:middle}}
      .f{{font-size:10px;fill:#475569;text-anchor:middle}}
      .b{{font-size:10px;font-weight:600;fill:#334155;text-anchor:middle}}
      .s{{font-size:8px;fill:#64748b;text-anchor:middle}}
      .l{{font-size:8px;fill:#334155;text-anchor:start}}
      .box{{rx:5;stroke-width:1.2}}
      .arr{{stroke:#475569;stroke-width:1.2;fill:none;marker-end:url(#rs_a)}}
      .res{{stroke:#ea580c;stroke-width:1.3;fill:none;stroke-dasharray:5 3}}
      .hi{{fill:#fff7ed;stroke:#ea580c;stroke-width:1.3}}
      .note{{fill:#f8fafc;stroke:#e2e8f0;stroke-width:1}}
    </style>
  </defs>

  <text x="430" y="24" class="t">Step 7-8: ShortConv residual and backbone writeback</text>
  <text x="430" y="44" class="f">
    <tspan font-style="italic">Y</tspan>
    <tspan> = SiLU(Conv1D(RMSNorm(</tspan>
    <tspan font-style="italic">V</tspan><tspan baseline-shift="super" font-size="0.65em">~</tspan>
    <tspan>))) + </tspan>
    <tspan font-style="italic">V</tspan><tspan baseline-shift="super" font-size="0.65em">~</tspan>
  </text>

  <!-- input V~ -->
  <rect x="24" y="118" width="64" height="40" class="box hi"/>
  <text x="56" y="136" class="b">
    <tspan font-style="italic">V</tspan><tspan baseline-shift="super" font-size="0.65em">~</tspan>
  </text>
  <text x="56" y="150" class="s">from Step 6</text>

  <line x1="88" y1="138" x2="118" y2="138" class="arr"/>

  <!-- main branch -->
  <rect x="118" y="118" width="88" height="40" class="box" fill="#eff6ff" stroke="#3b82f6"/>
  <text x="162" y="136" class="b">RMSNorm</text>
  <text x="162" y="150" class="s">per-t only</text>

  <line x1="206" y1="138" x2="236" y2="138" class="arr"/>

  <rect x="236" y="118" width="96" height="40" class="box" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="284" y="132" class="b">Conv1D</text>
  <text x="284" y="146" class="s">RF 1-&gt;10 here</text>

  <line x1="332" y1="138" x2="362" y2="138" class="arr"/>

  <rect x="362" y="118" width="72" height="40" class="box" fill="#ede9fe" stroke="#7c3aed"/>
  <text x="398" y="136" class="b">SiLU</text>
  <text x="398" y="150" class="s">nonlinear</text>

  <line x1="434" y1="138" x2="464" y2="138" class="arr"/>

  <!-- sum -->
  <circle cx="480" cy="138" r="16" fill="#fefce8" stroke="#ca8a04" stroke-width="1.4"/>
  <text x="480" y="142" class="b" font-size="14">+</text>

  <!-- residual bypass -->
  <path d="M 56 158 L 56 210 L 480 210 L 480 154" class="res"/>
  <text x="268" y="224" class="s" fill="#c2410c">residual path: Y_t keeps at least v~_t</text>

  <line x1="496" y1="138" x2="526" y2="138" class="arr"/>

  <!-- Y -->
  <rect x="526" y="118" width="52" height="40" class="box" fill="#dcfce7" stroke="#16a34a"/>
  <text x="552" y="143" class="b" font-style="italic">Y</text>

  <line x1="578" y1="138" x2="608" y2="138" class="arr"/>

  <!-- Step 8 -->
  <rect x="608" y="108" width="112" height="60" class="box" fill="#f0fdf4" stroke="#22c55e"/>
  <text x="664" y="128" class="b">Step 8</text>
  <text x="664" y="144" class="s">
    <tspan font-style="italic">H</tspan><tspan baseline-shift="super" font-size="0.65em">(l)</tspan>
    <tspan> &lt;- </tspan>
    <tspan font-style="italic">H</tspan><tspan baseline-shift="super" font-size="0.65em">(l)</tspan>
    <tspan> + </tspan>
    <tspan font-style="italic">Y</tspan>
  </text>
  <text x="664" y="158" class="s">inject into backbone</text>

  <line x1="664" y1="168" x2="664" y2="198" class="arr"/>

  <!-- Attention -->
  <rect x="608" y="198" width="112" height="44" class="box" fill="#fef3c7" stroke="#d97706"/>
  <text x="664" y="218" class="b">Attention</text>
  <text x="664" y="232" class="s">global dependencies</text>

  <!-- annotations left -->
  <rect x="24" y="248" width="400" height="76" class="note" rx="6"/>
  <text x="40" y="268" class="l" font-weight="600">Step 7 (conv branch)</text>
  <text x="40" y="284" class="l">RF widens only inside Conv1D (4 taps, gap d=3)</text>
  <text x="40" y="300" class="l">RF_seq: 1 -&gt; 1+(w-1)d (default 10)</text>
  <text x="40" y="316" class="l">RMSNorm / SiLU do not mix other positions</text>

  <rect x="436" y="248" width="400" height="76" class="note" rx="6"/>
  <text x="452" y="268" class="l" font-weight="600">Division of labor</text>
  <text x="452" y="284" class="l">Engram: local static n-gram memory (filtered)</text>
  <text x="452" y="300" class="l">Attention / MoE: dynamic reasoning after injection</text>
  <text x="452" y="316" class="l">residual ensures no loss of per-token v~_t</text>
</svg>
'''
    OUT.write_text(s, encoding="utf-8")
    try:
        root = ET.parse(OUT).getroot()
        if not root.attrib.get("viewBox"):
            raise SystemExit(["missing viewBox"])
    except ET.ParseError as e:
        raise SystemExit([str(e)])
    print("OK", OUT)


if __name__ == "__main__":
    main()
