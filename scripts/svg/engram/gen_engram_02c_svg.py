#!/usr/bin/env python3
"""Generate engram-02c-cxl-cache-access.svg with CXL vs RDMA pattern labels."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
from _paths import ENGRAM_DIAGRAMS  # noqa: E402

import xml.etree.ElementTree as ET
from pathlib import Path

OUT = ENGRAM_DIAGRAMS / "engram-02c-cxl-cache-access.svg"
W, H = 920, 580


def main() -> None:
    s = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="system-ui, -apple-system, sans-serif">
  <defs>
    <marker id="acc_a" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 Z" fill="#475569"/>
    </marker>
    <marker id="acc_o" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 Z" fill="#d97706"/>
    </marker>
    <marker id="acc_r" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto">
      <path d="M0,0 L7,3 L0,6 Z" fill="#94a3b8"/>
    </marker>
    <style>
      .title{{font-size:16px;font-weight:700;fill:#0f172a}}
      .sub{{font-size:11px;fill:#64748b}}
      .step{{font-size:10px;font-weight:700;fill:#fff}}
      .l{{font-size:10px;fill:#334155;text-anchor:middle}}
      .l0{{font-size:9px;fill:#64748b;text-anchor:middle}}
      .lh{{font-size:10px;font-weight:700;fill:#334155;text-anchor:middle}}
      .tag-cxl{{font-size:8px;font-weight:700;fill:#c2410c}}
      .tag-rdma{{font-size:8px;font-weight:700;fill:#64748b}}
      .box{{rx:8;stroke-width:1.5}}
      .cpu{{fill:#ecfdf5;stroke:#16a34a}}
      .gpu{{fill:#dbeafe;stroke:#2563eb}}
      .cxl{{fill:#fef3c7;stroke:#d97706}}
      .rdma{{fill:#f8fafc;stroke:#94a3b8;stroke-dasharray:4 3}}
      .arr{{stroke:#475569;stroke-width:1.4;fill:none;marker-end:url(#acc_a)}}
      .ov{{stroke:#d97706;stroke-width:1.5;fill:none;stroke-dasharray:5 3;marker-end:url(#acc_o)}}
      .rdma-arr{{stroke:#94a3b8;stroke-width:1.2;fill:none;stroke-dasharray:3 2;marker-end:url(#acc_r)}}
    </style>
  </defs>

  <text x="24" y="30" class="title">CXL Engram Cache Access Logic (one Decode step)</text>
  <text x="24" y="48" class="sub">5 KB/token/layer | deterministic index | prefetch overlaps layers 1..k-1</text>

  <!-- Step 0 -->
  <circle cx="70" cy="100" r="14" fill="#16a34a"/>
  <text x="70" y="104" class="step">0</text>
  <rect x="100" y="78" width="150" height="44" class="box cpu"/>
  <text x="175" y="96" class="l">CPU: hash offsets[]</text>
  <text x="175" y="112" class="l0">from token IDs only</text>

  <!-- Step 1 -->
  <circle cx="70" cy="170" r="14" fill="#16a34a"/>
  <text x="70" y="174" class="step">1</text>
  <rect x="100" y="148" width="150" height="44" class="box cpu"/>
  <text x="175" y="166" class="l">SGLang prefetch</text>
  <text x="175" y="182" class="l0">async on ForwardBatch</text>

  <!-- Step 2 -->
  <circle cx="70" cy="260" r="14" fill="#16a34a"/>
  <text x="70" y="264" class="step">2</text>
  <rect x="100" y="218" width="160" height="84" class="box cxl"/>
  <text x="180" y="238" class="l">L3 CXL Pool</text>
  <text x="180" y="254" class="l0">mmap: cxl_base+offset[i]</text>
  <text x="180" y="268" class="l0">discrete 320B rows</text>
  <text x="180" y="282" class="tag-cxl" text-anchor="middle">CXL.mem: load/store</text>
  <text x="180" y="294" class="l0">(cache-line, not RDMA msg)</text>

  <rect x="300" y="228" width="140" height="64" class="box cpu"/>
  <text x="370" y="248" class="l">path A</text>
  <text x="370" y="264" class="l0">OpenMP memcpy</text>
  <text x="370" y="278" class="tag-cxl" text-anchor="middle">CXL-&gt;CPU-&gt;GPU</text>

  <rect x="480" y="228" width="150" height="64" class="box gpu"/>
  <text x="555" y="248" class="l">L1 GPU staging</text>
  <text x="555" y="264" class="l0">gpu_staging[i]</text>
  <text x="555" y="278" class="tag-cxl" text-anchor="middle">cxl2vram_copy</text>

  <line x1="260" y1="260" x2="298" y2="260" class="arr"/>
  <line x1="260" y1="246" x2="478" y2="246" class="ov"/>
  <text x="368" y="238" class="tag-cxl">path B (preferred): CXL load/store + PCIe P2P</text>
  <line x1="440" y1="260" x2="478" y2="260" class="arr"/>

  <!-- Step 3 -->
  <circle cx="70" cy="350" r="14" fill="#16a34a"/>
  <text x="70" y="354" class="step">3</text>
  <rect x="100" y="328" width="530" height="44" class="box gpu"/>
  <text x="365" y="346" class="l">L1 Engram layer: gate / conv / residual + hidden</text>
  <text x="365" y="362" class="l0">then Attention -&gt; MoE -&gt; next layer</text>
  <line x1="555" y1="292" x2="365" y2="328" class="arr"/>

  <!-- Step 4 -->
  <circle cx="70" cy="420" r="14" fill="#94a3b8"/>
  <text x="70" y="424" class="step">4</text>
  <rect x="100" y="398" width="530" height="44" rx="8" fill="#f1f5f9" stroke="#94a3b8"/>
  <text x="365" y="416" class="l">optional Zipf hot-cache (L1/L2 hit skips CXL round-trip)</text>
  <text x="365" y="432" class="l0">miss: L3 read then optional L2 fill-back</text>

  <!-- prefetch window -->
  <rect x="660" y="78" width="230" height="188" rx="10" fill="#fff" stroke="#e2e8f0"/>
  <text x="775" y="100" class="lh">prefetch window</text>
  <text x="675" y="122" class="l" text-anchor="start" font-size="9">GPU: Layer 0 .. k-1</text>
  <rect x="675" y="128" width="200" height="16" fill="#2563eb" opacity="0.3"/>
  <text x="675" y="158" class="l" text-anchor="start" font-size="9">CXL: fetch @ layer k</text>
  <rect x="700" y="164" width="170" height="12" fill="#d97706" opacity="0.35"/>
  <text x="675" y="188" class="l" text-anchor="start" font-size="9">GPU: Layer k uses staging</text>
  <text x="775" y="214" class="l0">L_pool &lt; sum t_exec(1..k-1)</text>
  <text x="775" y="230" class="l0">e.g. layer-2 window ~56 us</text>
  <text x="775" y="252" class="l0" fill="#c2410c">needs ~DRAM latency</text>

  <line x1="250" y1="122" x2="658" y2="140" class="ov"/>
  <line x1="250" y1="170" x2="658" y2="170" class="ov"/>

  <!-- Communication pattern panel -->
  <rect x="660" y="280" width="230" height="248" rx="10" fill="#fff" stroke="#e2e8f0"/>
  <text x="775" y="302" class="lh">communication pattern</text>

  <!-- CXL box -->
  <rect x="672" y="314" width="206" height="108" rx="6" fill="#fff7ed" stroke="#ea580c" stroke-width="1.3"/>
  <text x="775" y="332" class="lh" fill="#c2410c">CXL (this diagram)</text>
  <text x="682" y="350" class="l" text-anchor="start" font-size="8">semantics: byte load/store on mmap</text>
  <text x="682" y="364" class="l" text-anchor="start" font-size="8">path: L3 cxl_base+off[i]</text>
  <text x="692" y="378" class="tag-cxl" text-anchor="start">--load/store--&gt;</text>
  <text x="682" y="392" class="l" text-anchor="start" font-size="8">PCIe P2P or CPU memcpy</text>
  <text x="692" y="406" class="tag-cxl" text-anchor="start">--&gt; L1 gpu_staging</text>
  <text x="775" y="416" class="l0" fill="#15803d">latency ~ DRAM | OK in 56 us</text>

  <!-- RDMA box -->
  <rect x="672" y="432" width="206" height="88" rx="6" class="rdma"/>
  <text x="775" y="450" class="lh" fill="#64748b">RDMA pool (KV-style)</text>
  <text x="682" y="468" class="l" text-anchor="start" font-size="8">semantics: get/put messages</text>
  <text x="682" y="482" class="l" text-anchor="start" font-size="8">path: GPU-&gt;bounce-&gt;NIC RDMA</text>
  <text x="692" y="496" class="tag-rdma" text-anchor="start">--get/put--&gt; remote DRAM</text>
  <text x="775" y="512" class="l0" fill="#b45309">many 320B pkts | too slow here</text>

  <!-- mini RDMA path sketch -->
  <rect x="682" y="456" width="28" height="14" rx="2" fill="#dbeafe" stroke="#2563eb" stroke-width="0.8"/>
  <text x="696" y="466" font-size="6" text-anchor="middle" fill="#2563eb">GPU</text>
  <line x1="710" y1="463" x2="722" y2="463" class="rdma-arr"/>
  <rect x="722" y="456" width="24" height="14" rx="2" fill="#f1f5f9" stroke="#94a3b8" stroke-width="0.8"/>
  <text x="734" y="466" font-size="5" text-anchor="middle" fill="#64748b">NIC</text>
  <line x1="746" y1="463" x2="758" y2="463" class="rdma-arr"/>
  <rect x="758" y="456" width="36" height="14" rx="2" fill="#f1f5f9" stroke="#94a3b8" stroke-width="0.8"/>
  <text x="776" y="466" font-size="5" text-anchor="middle" fill="#64748b">remote</text>

  <text x="460" y="568" class="l0">Orange dashed = CXL path B on main flow; grey panel = RDMA pattern Engram does NOT use</text>
</svg>
'''
    OUT.write_text(s, encoding="utf-8")
    ET.parse(OUT)
    print("OK", OUT)


if __name__ == "__main__":
    main()
