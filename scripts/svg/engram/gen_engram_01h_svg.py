#!/usr/bin/env python3
"""Generate engram-01h-rf-change-impact.svg (Chinese labels)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
from _paths import ENGRAM_DIAGRAMS  # noqa: E402

import xml.etree.ElementTree as ET
from pathlib import Path

OUT = ENGRAM_DIAGRAMS / "engram-01h-rf-change-impact.svg"
W, H = 920, 448

M, CW, GAP = 28, 420, 24
CL, CR = M, M + CW + GAP
CXL, CXR = CL + CW // 2, CR + CW // 2
PAD = 18


def main() -> None:
    s = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Noto Sans CJK SC, sans-serif">
  <defs>
    <style>
      .t{{font-size:15px;font-weight:700;fill:#0f172a;text-anchor:middle;font-family:"Noto Sans CJK SC",sans-serif}}
      .s{{font-size:9px;fill:#64748b;text-anchor:middle}}
      .rh{{font-size:10px;font-weight:700;fill:#334155;text-anchor:middle}}
      .tag{{font-size:8px;font-weight:600;text-anchor:middle}}
      .bl{{font-size:8.5px;fill:#334155;text-anchor:start}}
    </style>
  </defs>

  <text x="{W//2}" y="26" class="t">感受野 1 -&gt; 10：变了什么 / 没变什么</text>
  <text x="{W//2}" y="44" class="s">门控记忆流 V~（Step 6 之后；默认 w=4, d=3, RF_seq=10）</text>

  <rect x="{CL}" y="54" width="{CW}" height="22" rx="5" fill="#f1f5f9" stroke="#cbd5e1"/>
  <rect x="{CR}" y="54" width="{CW}" height="22" rx="5" fill="#f1f5f9" stroke="#cbd5e1"/>
  <text x="{CXL}" y="69" class="tag" fill="#475569">Step 7 之前</text>
  <text x="{CXR}" y="69" class="tag" fill="#475569">Step 7 之后（短卷积）</text>
  <text x="{W//2}" y="69" class="tag" fill="#7c3aed">RF_seq: 1 -&gt; 10</text>

  <text x="{CXL}" y="96" class="rh">1. 能力（训练+推理共有）</text>
  <rect x="{CL}" y="104" width="{CW}" height="76" rx="8" fill="#faf5ff" stroke="#d8b4fe"/>
  <rect x="{CR}" y="104" width="{CW}" height="76" rx="8" fill="#faf5ff" stroke="#d8b4fe"/>
  <text x="{CL+PAD}" y="124" class="bl">输入：每位置只有 v~_t</text>
  <text x="{CL+PAD}" y="140" class="bl">表达：逐 token 独立记忆卡</text>
  <text x="{CL+PAD}" y="156" class="bl">不能跨位置组合</text>
  <text x="{CL+PAD}" y="172" class="bl" fill="#b45309">短语级静态先验：弱</text>
  <text x="{CR+PAD}" y="124" class="bl">输入：v~_t, v~_t-3, v~_t-6, v~_t-9</text>
  <text x="{CR+PAD}" y="140" class="bl">表达：记忆流上的局部短语</text>
  <text x="{CR+PAD}" y="156" class="bl">SiLU(Conv) 非线性混合</text>
  <text x="{CR+PAD}" y="172" class="bl" fill="#15803d">仍局部；全局靠 Attention</text>

  <text x="{CXL}" y="196" class="rh">2. 训练</text>
  <rect x="{CL}" y="204" width="{CW}" height="56" rx="8" fill="#eff6ff" stroke="#bfdbfe"/>
  <rect x="{CR}" y="204" width="{CW}" height="56" rx="8" fill="#eff6ff" stroke="#bfdbfe"/>
  <text x="{CL+PAD}" y="224" class="bl">梯度只回传到位置 t</text>
  <text x="{CL+PAD}" y="240" class="bl">各 token 记忆路径解耦</text>
  <text x="{CL+PAD}" y="256" class="bl">无可学习的跨位置混合</text>
  <text x="{CR+PAD}" y="224" class="bl">梯度耦合 t, t-3, t-6, t-9</text>
  <text x="{CR+PAD}" y="240" class="bl">W_conv 学习如何组合记忆</text>
  <text x="{CR+PAD}" y="256" class="bl">代价：O(w) depthwise conv</text>

  <text x="{CXL}" y="276" class="rh">3. 推理</text>
  <rect x="{CL}" y="284" width="{CW}" height="56" rx="8" fill="#ecfdf5" stroke="#86efac"/>
  <rect x="{CR}" y="284" width="{CW}" height="56" rx="8" fill="#ecfdf5" stroke="#86efac"/>
  <text x="{CL+PAD}" y="304" class="bl">无额外 Conv</text>
  <text x="{CL+PAD}" y="320" class="bl">不需 v~ 历史缓存</text>
  <text x="{CL+PAD}" y="336" class="bl" fill="#64748b">略快（跳过 Step 7）</text>
  <text x="{CR+PAD}" y="304" class="bl">查表仍 O(1)/token</text>
  <text x="{CR+PAD}" y="320" class="bl">多 w=4 次 conv（非 10 次查表）</text>
  <text x="{CR+PAD}" y="336" class="bl">缓存约 9 个 v~；prefetch 不变</text>

  <text x="{W//2}" y="360" class="rh">4. 不变（RF 扩大不碰这些）</text>
  <rect x="{M}" y="368" width="{W-2*M}" height="48" rx="8" fill="#f8fafc" stroke="#e2e8f0"/>
  <text x="{M+PAD}" y="388" class="bl">n-gram 查表 O(1)，索引只依赖 input ids</text>
  <text x="{M+PAD}" y="404" class="bl">Step 6 Gate 仍先过滤 v~_t</text>
  <text x="480" y="388" class="bl">全局依赖仍由 Step 8 后 Attention/MoE 负责</text>
  <text x="480" y="404" class="bl" fill="#15803d">RF=10 是 V~ 混合变宽，不是 10 倍带宽</text>
</svg>
'''
    OUT.write_text(s, encoding="utf-8")
    ET.parse(OUT)
    print("OK", OUT)


if __name__ == "__main__":
    main()
