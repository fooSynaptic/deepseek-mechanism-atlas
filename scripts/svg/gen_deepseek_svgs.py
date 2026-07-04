#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate SVG diagrams for deepseek-ai docs."""
from __future__ import annotations

from pathlib import Path

import sys

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import DIAGRAMS, FIGURES, REPO  # noqa: E402

import html
import os
import xml.etree.ElementTree as ET
from pathlib import Path

OUT = str(DIAGRAMS)


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def svg_header(w: int, h: int, prefix: str = "ds") -> str:
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">
<style><![CDATA[
  .t {{ font: 700 15px sans-serif; fill: #1a1a2e; text-anchor: middle; }}
  .st {{ font: 11px sans-serif; fill: #666; text-anchor: middle; }}
  .ph {{ font: 600 12px sans-serif; fill: #4a5a7a; }}
  .lb {{ font: 600 11px sans-serif; fill: #222; text-anchor: middle; }}
  .dt {{ font: 10px sans-serif; fill: #555; text-anchor: middle; }}
  .an {{ font: 9px sans-serif; fill: #2563eb; text-anchor: middle; }}
  .arr {{ stroke: #666; stroke-width: 1.5; fill: none; marker-end: url(#{prefix}a); }}
  .arr-d {{ stroke: #888; stroke-width: 1.2; fill: none; stroke-dasharray: 5 4; marker-end: url(#{prefix}ad); }}
  .arr-b {{ stroke: #4A90D9; stroke-width: 1.5; fill: none; marker-end: url(#{prefix}ab); }}
  .arr-g {{ stroke: #27AE60; stroke-width: 1.5; fill: none; marker-end: url(#{prefix}ag); }}
]]></style>
<defs>
  <marker id="{prefix}a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#666"/></marker>
  <marker id="{prefix}ad" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#888"/></marker>
  <marker id="{prefix}ab" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#4A90D9"/></marker>
  <marker id="{prefix}ag" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#27AE60"/></marker>
</defs>
'''


def box(x, y, w, h, fill="#f8f9fc", stroke="#c5cee0", rx=6):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
    )


def phase(x, y, w, h, label, stroke="#b0bdd4"):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#eef1f8" '
        f'stroke="{stroke}" stroke-width="1.5" stroke-dasharray="6 3"/>'
        f'<text x="{x + 12}" y="{y + 18}" class="ph">{esc(label)}</text>'
    )


def txt(x, y, s, cls="lb"):
    return f'<text x="{x}" y="{y}" class="{cls}">{esc(s)}</text>'


def line(x1, y1, x2, y2, cls="arr"):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="{cls}"/>'


def node(cx, cy, w, h, title, lines, fill="#f8f9fc", stroke="#c5cee0"):
    """Centered box; title + detail lines with fixed vertical rhythm."""
    x, y = cx - w // 2, cy - h // 2
    s = box(x, y, w, h, fill, stroke)
    s += txt(cx, y + 18, title)
    for i, ln in enumerate(lines):
        s += txt(cx, y + 34 + i * 14, ln, "dt")
    return s, x, y, x + w, y + h


def node_auto(cx, cy, w, title, lines, fill="#f8f9fc", stroke="#c5cee0"):
    """Box height from line count (avoids overlapping dt rows)."""
    h = max(56, 36 + len(lines) * 14)
    return node(cx, cy, w, h, title, lines, fill, stroke)


def write(name: str, body: str, w: int, h: int, prefix: str = "ds", out_dir: str | None = None) -> None:
    content = svg_header(w, h, prefix) + body + "</svg>"
    base = out_dir or OUT
    p = os.path.join(base, name)
    os.makedirs(base, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(content)
    ET.parse(p)
    import sys

    sys.path.insert(0, OUT)
    from svg_validate import validate_svg  # noqa: E402

    layout_errs = validate_svg(Path(p))
    if layout_errs:
        raise SystemExit(f"FAIL {name}: {', '.join(layout_errs)}")
    print("OK", p)


def gen_version_quick() -> None:
    """README horizontal lineage: V3 -> V3.1 -> V3.2 -> V4; Index Share patch below V3.2."""
    w, h = 920, 260
    s = ""
    s += txt(460, 22, "DeepSeek 版本快速对照", "t")
    s += txt(460, 40, "主链: V3 -> V3.1 -> V3.2 -> V4  |  Index Share 为 V3.2 纯 infra 补丁", "st")

    bw, bh = 128, 64
    y_main = 68
    cy = y_main + bh // 2

    main = [
        (100, "DeepSeek-V3", ["MoE + MLA", "128K"], "#f8f9fc", "#c5cee0"),
        (280, "V3.1 / Terminus", ["Hybrid 推理", "128K"], "#f8f9fc", "#c5cee0"),
        (460, "V3.2", ["+ DSA 稀疏注意力"], "#eef4fc", "#4A90D9"),
        (780, "V4", ["CSA + HCA + mHC", "1M context"], "#fff8ee", "#E67E22"),
    ]
    for cx, title, lines, fill, stroke in main:
        s += node_auto(cx, cy, bw, title, lines, fill, stroke)[0]

    for i in range(len(main) - 1):
        x1 = main[i][0] + bw // 2
        x2 = main[i + 1][0] - bw // 2
        s += line(x1, cy, x2, cy)

    patch_y = 168
    patch_cy = patch_y + bh // 2
    s += node_auto(460, patch_cy, bw, "Index Share", ["IndexCache", "跨层复用"], "#f0faf0", "#27AE60")[0]
    s += line(460, y_main + bh, 460, patch_y, "arr-g")
    s += txt(460, patch_y + bh + 20, "纯 infra 补丁", "an")

    write("deepseek-version-quick.svg", s, w, h, "dsq")


def gen_version_lineage() -> None:
  """Report full lineage: algorithm lane + infra patches."""
  w, h = 920, 448
  s = ""
  s += txt(460, 22, "DeepSeek 版本时间线与关系", "t")
  s += txt(460, 40, "算法演进 (上) 与基础设施补丁 (下)", "st")

  algo_phase_y, algo_phase_h = 56, 228
  s += phase(24, algo_phase_y, 872, algo_phase_h, "算法演进")

  bw = 108
  algo_box_h = 78
  algo_box_y = 103
  algo_y = algo_box_y + algo_box_h // 2
  algo = [
      (78, "V3", ["2024-12", "MoE + MLA", "128K"]),
      (198, "R1", ["2025-01", "V3-Base", "+ RLVR"]),
      (318, "V3.1-T", ["2025", "Hybrid", "128K"]),
      (438, "V3.2-Exp", ["DeepSeek", "2025-09", "+ DSA 实验"]),
      (558, "V3.2", ["2025-12", "DSA 正式版"]),
      (698, "V4", ["2026", "CSA + HCA + mHC", "1M"]),
  ]
  algo_boxes: list[tuple[int, int, int, int]] = []
  for cx, title, lines in algo:
      frag, x0, y0, x1, y1 = node(cx, algo_y, bw, algo_box_h, title, lines, "#f8f9fc", "#c5cee0")
      s += frag
      algo_boxes.append((x0, y0, x1, y1))

  # Main chain: V3.1-T -> V3.2-Exp -> V3.2 -> V4
  main_chain = [2, 3, 4, 5]
  for i in range(len(main_chain) - 1):
      a, b = main_chain[i], main_chain[i + 1]
      x1 = algo[a][0] + bw // 2
      x2 = algo[b][0] - bw // 2
      s += line(x1, algo_y, x2, algo_y)

  # R1 branch above boxes (V3 -> R1 -> V3.1-T)
  branch_y = algo_boxes[0][1] - 18
  s += line(algo[0][0], branch_y, algo[1][0], branch_y, "arr-b")
  s += line(algo[1][0] + bw // 2, branch_y, algo[2][0] - bw // 2, algo_box_y + 8, "arr-b")
  s += txt(248, branch_y - 10, "post-train / RLVR", "an")

  infra_phase_y, infra_phase_h = 300, 132
  s += phase(24, infra_phase_y, 872, infra_phase_h, "基础设施补丁", "#8fbc8f")

  infra_y = 362
  infra = [
      (180, "Index Share", ["IndexCache", "V3.2 衍生"]),
      (360, "ESS", ["Latent offload", "V3.2 衍生"]),
      (540, "DSpark", ["V4 投机解码", "DeepSpec"]),
      (720, "HiSparse", ["V4 异构 KV", "CPU + 磁盘"]),
  ]
  for cx, title, lines in infra:
      s += node_auto(cx, infra_y, 130, title, lines, "#f0faf0", "#27AE60")[0]

  v32_bottom = algo_boxes[4][3]
  v4_bottom = algo_boxes[5][3]
  infra_top = infra_y - 40
  s += line(558, v32_bottom, 180, infra_top, "arr-g")
  s += line(558, v32_bottom, 360, infra_top, "arr-g")
  s += line(698, v4_bottom, 540, infra_top, "arr-g")
  s += line(698, v4_bottom, 720, infra_top, "arr-g")
  s += txt(400, infra_phase_y - 8, "V3.2 衍生", "an")
  s += txt(620, infra_phase_y - 8, "V4 衍生", "an")

  write("deepseek-version-lineage.svg", s, w, h, "dsl")


def gen_bbpe_process_example() -> None:
    """BBPE training walkthrough: text -> UTF-8 bytes -> iterative merge -> subwords."""
    w, h = 920, 560
    pf = "bbpe"
    fig_dir = os.path.join(os.path.dirname(OUT), "docs", "figures", "v1", "bbpe")
    s = ""
    s += txt(460, 24, "BBPE 训练过程示例 (示意)", "t")
    s += txt(460, 42, "DeepSeek V1: UTF-8 byte-level BPE + 100k merges", "st")

    # --- Step 1: input ---
    s += phase(40, 58, 840, 88, "Step 1  Pre-tokenization + UTF-8")
    s += node_auto(190, 108, 160, "Input text", ["Hi + zh sample"], "#fff8ee", "#E67E22")[0]
    s += node_auto(530, 108, 320, "UTF-8 byte sequence", [
        "48 69  (Hi)",
        "E4 B8 96  (char A)",
        "E7 95 8C  (char B)",
    ], "#f8f9fc", "#c5cee0")[0]
    s += line(280, 108, 360, 108, "arr-b")

    # --- Step 2: base vocab ---
    s += phase(40, 162, 840, 72, "Step 2  Base symbols (256 bytes)")
    s += node_auto(460, 200, 520, "Initial tokens = single bytes", [
        "Vocab seed: 0x00 .. 0xFF (256 entries)",
        "Any UTF-8 string decomposes to bytes -> no UNK at base layer",
    ], "#eef4fc", "#4A90D9")[0]

    # --- Step 3: merge iterations ---
    s += phase(40, 250, 840, 168, "Step 3  Iterative merge (frequency-driven)")
    iter_y = 318
    bw = 240
    iters = [
        (180, "Merge 1", [
            "Pair (E4, B8) most frequent",
            "New token T1 = E4+B8",
            "Seq: 48 69 | T1 96 | ...",
        ], "#f8f9fc", "#c5cee0"),
        (460, "Merge 2", [
            "Pair (T1, 96) -> T2",
            "T2 encodes one zh char",
            "Seq: 48 69 | T2 | ...",
        ], "#f8f9fc", "#c5cee0"),
        (740, "Merge 3", [
            "Pair (48, 69) -> T3",
            "T3 = Hi subword",
            "Repeat until 100k merges",
        ], "#f8f9fc", "#c5cee0"),
    ]
    for cx, title, lines, fill, stroke in iters:
        s += node_auto(cx, iter_y, bw, title, lines, fill, stroke)[0]
    s += line(180 + bw // 2, iter_y, 460 - bw // 2, iter_y)
    s += line(460 + bw // 2, iter_y, 740 - bw // 2, iter_y)
    s += txt(460, 392, "Each step: count adjacent pair freq, merge max, append to merge table", "an")

    # --- Step 4: encode + vocab ---
    s += phase(40, 432, 840, 108, "Step 4  Encode (inference) + final vocab")
    s += node_auto(260, 488, 220, "Greedy longest match", [
        "Apply merge table",
        "Hi -> T3, zh -> T2/T4",
        "Output token id sequence",
    ], "#f0faf0", "#27AE60")[0]
    s += node_auto(660, 488, 280, "V1 BBPE vocabulary", [
        "256 bytes + 100,000 merges",
        "+ 15 special -> 100,015 ids",
        "embedding rows: 102,400 (reserved)",
    ], "#f0faf0", "#27AE60")[0]
    s += line(370, 488, 520, 488, "arr-g")

    write(
        "bbpe-process-example.svg",
        s,
        w,
        h,
        pf,
        out_dir=fig_dir,
    )


def main() -> None:
  gen_version_quick()
  gen_version_lineage()
  gen_bbpe_process_example()
  from gen_opt_direction_ternary_svg import gen_opt_direction_ternary  # noqa: E402

  gen_opt_direction_ternary()
  from gen_v3_moe_vs_v2_svg import gen_v3_moe_vs_v2  # noqa: E402
  from gen_grpo_vs_ppo_svg import gen_grpo_vs_ppo  # noqa: E402
  from gen_mtp_speculative_svg import gen_mtp_speculative  # noqa: E402
  from gen_mla_mode_switch_svg import gen_mla_mode_switch  # noqa: E402
  from gen_v3_fp8_quant_svg import gen_v3_fp8_quant  # noqa: E402
  from gen_mla_forward_flow_svg import gen_mla_forward_flow  # noqa: E402
  from gen_v4_hetero_kv_svg import gen_v4_hetero_kv  # noqa: E402
  from gen_dspark_speculative_svg import gen_dspark_speculative  # noqa: E402

  gen_v3_moe_vs_v2()
  gen_grpo_vs_ppo()
  gen_mtp_speculative()
  gen_mla_mode_switch()
  gen_v3_fp8_quant()
  gen_mla_forward_flow()
  gen_v4_hetero_kv()
  gen_dspark_speculative()


if __name__ == "__main__":
  main()
