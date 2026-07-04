#!/usr/bin/env python3
"""Generate engram-01f-rf10-train-infer.svg with aligned 2-column grid."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
from _paths import ENGRAM_DIAGRAMS  # noqa: E402

import xml.etree.ElementTree as ET
from pathlib import Path

OUT = ENGRAM_DIAGRAMS / "engram-01f-rf10-train-infer.svg"
W, H = 920, 432

# symmetric 2-col grid: margin | col | gap | col | margin
MARGIN = 28
COL_W = 420
GAP = 24
COL_L = MARGIN
COL_R = MARGIN + COL_W + GAP
CX_L = COL_L + COL_W // 2
CX_R = COL_R + COL_W // 2
PAD = 18  # inner text padding from box left edge


def main() -> None:
    # token strip: 4 boxes centered in column
    tok_w, tok_h, tok_gap = 36, 28, 14
    tok_total = 4 * tok_w + 3 * tok_gap
    tok_x0_l = COL_L + (COL_W - tok_total) // 2
    tok_x0_r = COL_R + (COL_W - tok_total) // 2

    def tok_boxes(x0: int, y: int, active_fill: str, active_stroke: str) -> str:
        labels = ["v~_t-9", "v~_t-6", "v~_t-3", "v~_t"]
        s = ""
        for i, lab in enumerate(labels):
            x = x0 + i * (tok_w + tok_gap)
            hi = i == 3
            fill = "#fff7ed" if hi else active_fill
            stroke = "#ea580c" if hi else active_stroke
            sw = "1.3" if hi else "1.2"
            s += f'  <rect x="{x}" y="{y}" width="{tok_w}" height="{tok_h}" class="box" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>\n'
            s += f'  <text x="{x + tok_w // 2}" y="{y + 18}" class="tc">{lab}</text>\n'
        return s

    train_lines = [
        ("+", "Gradients flow to t, t-3, t-6, t-9 via conv weights", "#334155"),
        ("+", "Learns how to combine filtered memories (not fixed concat)", "#334155"),
        ("+", "SiLU nonlinearity on memory stream", "#334155"),
        ("=", "Cost: O(w) depthwise conv per token; cheap vs Attention", "#334155"),
        ("=", "Causal: no future token in loss", "#334155"),
        ("", "Without Step 7: per-token memory decoupled, weaker phrases", "#b45309"),
    ]
    infer_lines = [
        ("+", "Lookup still O(1) per token (Step 3 unchanged)", "#334155"),
        ("+", "Extra: w=4 conv taps per step (not 10x table reads)", "#334155"),
        ("+", "Cache last ~9 v~ for dilated conv (small vs KV)", "#334155"),
        ("+", "Prefetch/offload path unaffected", "#334155"),
        ("=", "Global context still from Attention after Engram", "#334155"),
        ("", "Latency: conv usually minor vs memory lookup", "#15803d"),
    ]

    def bullet_block(x_box: int, y_box: int, lines: list[tuple[str, str, str]]) -> str:
        tx = x_box + PAD
        s = ""
        for i, (prefix, text, color) in enumerate(lines):
            y = y_box + 22 + i * 17
            label = f"{prefix} {text}".strip() if prefix else text
            s += f'  <text x="{tx}" y="{y}" class="bl" fill="{color}">{label}</text>\n'
        return s

    s = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="system-ui, -apple-system, sans-serif">
  <defs>
    <marker id="rf01f_a" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#475569"/>
    </marker>
    <style>
      .t{{font-size:14px;font-weight:700;fill:#0f172a;text-anchor:middle}}
      .h{{font-size:11px;font-weight:600;fill:#334155;text-anchor:middle}}
      .s{{font-size:8px;fill:#64748b;text-anchor:middle}}
      .tc{{font-size:8px;fill:#334155;text-anchor:middle}}
      .bl{{font-size:9px;fill:#334155;text-anchor:start}}
      .box{{rx:5}}
    </style>
  </defs>

  <text x="{W // 2}" y="24" class="t">Step 7: RF 1 to 10 - Train vs Infer</text>
  <text x="{W // 2}" y="42" class="s">Gated memory stream V~ : sequence span 1 -&gt; 1+(w-1)d = 10 (default w=4, d=3)</text>

  <!-- row 1 left -->
  <text x="{CX_L}" y="66" class="h">A. After Step 6 (RF = 1)</text>
  <rect x="{COL_L}" y="74" width="{COL_W}" height="96" rx="8" fill="#f8fafc" stroke="#e2e8f0"/>
  <text x="{CX_L}" y="90" class="s">Each position uses only its own v~_t</text>
{tok_boxes(tok_x0_l, 100, "#dcfce7", "#16a34a")}  <text x="{COL_L + PAD}" y="148" class="bl" font-size="8" fill="#64748b">Y_t = v~_t only</text>
  <text x="{CX_L}" y="162" class="s">No cross-position mix on memory stream</text>

  <!-- row 1 right -->
  <text x="{CX_R}" y="66" class="h">B. After Step 7 (RF = 10)</text>
  <rect x="{COL_R}" y="74" width="{COL_W}" height="96" rx="8" fill="#f8fafc" stroke="#e2e8f0"/>
  <text x="{CX_R}" y="90" class="s">Dilated causal conv mixes 4 taps, span 10 indices</text>
{tok_boxes(tok_x0_r, 100, "#dbeafe", "#2563eb")}'''
    # conv lines on right panel
    yt = 114
    x_t = tok_x0_r + 3 * (tok_w + tok_gap) + tok_w // 2
    for i in range(3):
        x_s = tok_x0_r + i * (tok_w + tok_gap) + tok_w // 2
        s += f'  <line x1="{x_s}" y1="{yt}" x2="{x_t}" y2="{yt + 12}" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4 3" fill="none"/>\n'
    s += f'  <line x1="{x_t}" y1="{yt}" x2="{x_t}" y2="{yt + 12}" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4 3" fill="none"/>\n'
    y_out = tok_x0_r + 3 * (tok_w + tok_gap) + tok_w + 10
    s += f'  <rect x="{y_out}" y="104" width="40" height="24" class="box" fill="#ede9fe" stroke="#7c3aed" stroke-width="1.2"/>\n'
    s += f'  <text x="{y_out + 20}" y="120" class="tc">Y_t</text>\n'
    s += f'  <text x="{CX_R}" y="162" class="s">SiLU(Conv)+residual; phrase-level local memory</text>\n'

    # row 2
    s += f'''
  <text x="{CX_L}" y="188" class="h">Training impact</text>
  <rect x="{COL_L}" y="196" width="{COL_W}" height="112" rx="8" fill="#eff6ff" stroke="#bfdbfe"/>
{bullet_block(COL_L, 196, train_lines)}
  <text x="{CX_R}" y="188" class="h">Inference impact</text>
  <rect x="{COL_R}" y="196" width="{COL_W}" height="112" rx="8" fill="#ecfdf5" stroke="#86efac"/>
{bullet_block(COL_R, 196, infer_lines)}

  <!-- summary -->
  <rect x="{MARGIN}" y="324" width="{W - 2 * MARGIN}" height="56" rx="8" fill="#fafafa" stroke="#e2e8f0"/>
  <text x="{W // 2}" y="344" class="h">Summary</text>
  <text x="{W // 2}" y="360" class="s">RF 10 = wider dependency along V~, not 10x compute. Train: coupled grads + richer local memory.</text>
  <text x="{W // 2}" y="372" class="s">Infer: +4-tap conv + tiny v~ buffer. See qa/step7-rf10-train-infer-impact.md</text>
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
