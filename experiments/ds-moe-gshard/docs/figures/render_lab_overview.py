#!/usr/bin/env python3
"""Render lab overview + F/S ablation SVGs (English labels)."""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent


def write(name: str, body: str) -> None:
    path = OUT / name
    path.write_text(body.strip() + "\n")
    print(f"wrote {path}")


def overview() -> str:
    # Narrative (attribution / contrast / locked setup) lives in Markdown beside
    # the figure; keep this SVG metrics-only (see generate-svg-diagram skill).
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 310"
     font-family="Helvetica Neue,Arial,sans-serif">
<defs>
  <style>
    .title {{ font-size: 20px; font-weight: 700; fill: #0f172a; }}
    .sub {{ font-size: 12px; fill: #64748b; }}
    .h {{ font-size: 13px; font-weight: 700; fill: #1e293b; }}
    .t {{ font-size: 11.5px; fill: #475569; }}
    .num {{ font-size: 18px; font-weight: 700; }}
    .badge {{ font-size: 11px; font-weight: 600; fill: #fff; }}
    .ok {{ fill: #166534; }}
    .weak {{ fill: #a16207; }}
    .miss {{ fill: #b91c1c; }}
  </style>
</defs>
<rect width="100%" height="100%" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>
<text x="36" y="36" class="title">ds-moe-gshard scorecard (10B, data_l2, GPT-2 50k)</text>
<text x="36" y="56" class="sub">Primary metric: held-out val BPB (lower is better).</text>

<!-- E1 -->
<rect x="36" y="78" width="250" height="210" rx="10" fill="#fff" stroke="#bfdbfe"/>
<rect x="52" y="92" width="100" height="22" rx="11" fill="#2563eb"/>
<text x="102" y="107" text-anchor="middle" class="badge">E1 ranking</text>
<text x="52" y="140" class="h">Matched FLOP</text>
<text x="52" y="162" class="t">GShard Top-2: 0.7970</text>
<text x="52" y="182" class="t">DSMoE 1+Top-7: 0.7850</text>
<text x="52" y="214" class="num" fill="#dc2626">Δ −0.012</text>
<text x="52" y="238" class="t">Direction hit; C2 magnitude miss</text>
<text x="52" y="258" class="t weak">Gap ~ last-1B scatter (0.015)</text>
<text x="52" y="278" class="t ok">LB fix required for layout gap</text>

<!-- E3 -->
<rect x="302" y="78" width="250" height="210" rx="10" fill="#fff" stroke="#a7f3d0"/>
<rect x="318" y="92" width="120" height="22" rx="11" fill="#059669"/>
<text x="378" y="107" text-anchor="middle" class="badge">E3 GShard x1.5</text>
<text x="318" y="140" class="h">Matches x1.5 compute</text>
<text x="318" y="162" class="t">GShard x1.5: 0.7860</text>
<text x="318" y="182" class="t">vs DSMoE: |Δ| = 0.001</text>
<text x="318" y="214" class="num" fill="#059669">≈ DSMoE</text>
<text x="318" y="238" class="t ok">E3-A/B/C all hit</text>
<text x="318" y="258" class="t">Beats GShard 1.0x by 0.011</text>
<text x="318" y="278" class="t">Paper Table 2 direction</text>

<!-- F -->
<rect x="568" y="78" width="250" height="210" rx="10" fill="#fff" stroke="#fde68a"/>
<rect x="584" y="92" width="150" height="22" rx="11" fill="#d97706"/>
<text x="659" y="107" text-anchor="middle" class="badge">F half-activated</text>
<text x="584" y="140" class="h">1 shared + Top-3</text>
<text x="584" y="162" class="t">Train: 0.8014 (act 1.0x)</text>
<text x="584" y="182" class="t">vs GShard: +0.004</text>
<text x="584" y="214" class="num" fill="#a16207">≈ GShard</text>
<text x="584" y="238" class="t weak">Near-match only</text>
<text x="584" y="258" class="t">Eval k≥4 still beats GShard</text>
<text x="584" y="278" class="t">Half activated vs matched FLOP</text>

<!-- S -->
<rect x="834" y="78" width="250" height="210" rx="10" fill="#fff" stroke="#fecaca"/>
<rect x="850" y="92" width="150" height="22" rx="11" fill="#dc2626"/>
<text x="925" y="107" text-anchor="middle" class="badge">S shared ablation</text>
<text x="850" y="140" class="h">Eval surgery / train</text>
<text x="850" y="162" class="t">Eval surgery: +0.135 BPB</text>
<text x="850" y="182" class="t">Train 0+8: 0.7933</text>
<text x="850" y="214" class="num" fill="#7c3aed">+0.008 vs DSMoE</text>
<text x="850" y="238" class="t ok">S-A soft hit</text>
<text x="850" y="258" class="t">From-scratch gap &lt;&lt; surgery</text>
<text x="850" y="278" class="t">Slightly beats GShard (−0.004)</text>
</svg>'''

def ablation() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1100 560"
     font-family="Helvetica Neue,Arial,sans-serif">
<defs>
  <style>
    .title {{ font-size: 18px; font-weight: 700; fill: #0f172a; }}
    .sub {{ font-size: 12px; fill: #64748b; }}
    .h {{ font-size: 13px; font-weight: 700; fill: #1e293b; }}
    .t {{ font-size: 11.5px; fill: #475569; }}
    .badge {{ font-size: 11px; font-weight: 600; fill: #fff; }}
  </style>
  <marker id="arr" viewBox="0 0 10 8" refX="9" refY="4"
          markerWidth="8" markerHeight="6" orient="auto-start-reverse">
    <path d="M0,0 L10,4 L0,8 Z" fill="#64748b"/>
  </marker>
</defs>
<rect width="100%" height="100%" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>
<text x="36" y="34" class="title">§4.5 arms: F (fewer activated) vs S (drop shared)</text>
<text x="36" y="54" class="sub">Do not merge. F keeps shared. S removes shared and adds one routed slot.</text>

<!-- baseline -->
<rect x="380" y="80" width="340" height="120" rx="10" fill="#fff" stroke="#cbd5e1"/>
<rect x="396" y="94" width="140" height="22" rx="11" fill="#334155"/>
<text x="466" y="109" text-anchor="middle" class="badge">E1 DeepSeekMoE</text>
<text x="396" y="140" class="h">1 shared + Top-7 / 63</text>
<text x="396" y="162" class="t">activated = 2.0x dense FFN</text>
<text x="396" y="184" class="t">val BPB 0.7850 @ 10B train</text>

<!-- F -->
<rect x="36" y="260" width="480" height="260" rx="10" fill="#fff" stroke="#fde68a"/>
<rect x="52" y="274" width="70" height="22" rx="11" fill="#d97706"/>
<text x="87" y="289" text-anchor="middle" class="badge">Arm F</text>
<text x="52" y="320" class="h">Fewer activated (keep shared)</text>
<text x="52" y="344" class="t">F-eval (Fig 5): change k at eval on E1 ckpt</text>
<text x="52" y="366" class="t">k=4 already ≤ same-batch GShard → direction hit</text>
<text x="52" y="396" class="t">F-train (Fig 6): train 1+Top-3 from scratch</text>
<text x="52" y="418" class="t">activated = 1.0x dense (half of GShard)</text>
<text x="52" y="440" class="t">result 0.8014 vs GShard 0.7970 (Δ +0.004)</text>
<text x="52" y="470" class="t">Attribution: near-match / tokens (16x total experts retained)</text>
<text x="52" y="492" class="t">Collapse: no (max_frac ≈ 0.12)</text>

<!-- S -->
<rect x="584" y="260" width="480" height="260" rx="10" fill="#fff" stroke="#fecaca"/>
<rect x="600" y="274" width="70" height="22" rx="11" fill="#dc2626"/>
<text x="635" y="289" text-anchor="middle" class="badge">Arm S</text>
<text x="600" y="320" class="h">Drop shared (matched activated FLOPs)</text>
<text x="600" y="344" class="t">S-eval: n_shared=0, k_routed=8 on E1 ckpt</text>
<text x="600" y="366" class="t">0.8204 → 0.9558 (Δ +0.135 BPB)</text>
<text x="600" y="396" class="t">Large degradation is the surgery claim</text>
<text x="600" y="418" class="t">Shared not replaceable by +1 routed (eval)</text>
<text x="600" y="440" class="t">S-train: 0 shared + Top-8 from scratch → 0.7933</text>
<text x="600" y="462" class="t">vs DSMoE +0.008 (S-A hit); &lt;&lt; S-eval +0.135</text>
<text x="600" y="484" class="t">S-eval ≠ S-train; both closed</text>
<text x="600" y="506" class="t">Paper surgery analogue: Pile 1.808 → 2.414</text>

<!-- arrows -->
<line x1="470" y1="200" x2="220" y2="260" stroke="#64748b" stroke-width="1.5" marker-end="url(#arr)"/>
<line x1="630" y1="200" x2="820" y2="260" stroke="#64748b" stroke-width="1.5" marker-end="url(#arr)"/>
<text x="280" y="230" class="t" fill="#d97706">keep shared, cut k</text>
<text x="780" y="230" class="t" fill="#dc2626">drop shared, k=8</text>
</svg>'''


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write("lab_scorecard.svg", overview())
    write("ablation_fs_arms.svg", ablation())


if __name__ == "__main__":
    main()
