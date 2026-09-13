#!/usr/bin/env python3
"""Generate CED / CSA2 inference-process SVGs for V4.1-Flash notes (CJK as hex entities)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "diagrams"
FIG_DIR = ROOT / "docs" / "versions" / "figures"


def e(s: str) -> str:
    return "".join(f"&#x{ord(c):X};" if ord(c) > 127 else c for c in s)


def write_svg(name: str, lines: list[str], *, extra_dirs: list[Path] | None = None) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    text = "\n".join(lines) + "\n"
    path.write_text(text, encoding="utf-8")
    shutil.copy2(path, FIG_DIR / name)
    for d in extra_dirs or []:
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, d / name)
    subprocess.run(["xmllint", "--noout", str(path)], check=True)
    return path


def defs(extra_markers: list[tuple[str, str]] | None = None) -> list[str]:
    markers = [
        ("arrow-blue", "#3b82f6"),
        ("arrow-orange", "#ea580c"),
        ("arrow-violet", "#7c3aed"),
        ("arrow-slate", "#64748b"),
        ("arrow-teal", "#0d9488"),
    ]
    if extra_markers:
        markers.extend(extra_markers)
    L = [
        "<defs>",
        "  <style>",
        "    .title { font-size: 18px; font-weight: 700; fill: #1a1a2e; }",
        "    .subtitle { font-size: 11.5px; fill: #64748b; }",
        "    .badge-text { font-size: 11px; font-weight: 600; fill: #fff; }",
        "    .node-title { font-size: 12.5px; font-weight: 600; fill: #1e293b; }",
        "    .node-text { font-size: 11px; fill: #475569; }",
        "    .mono { font-family: Consolas,Menlo,monospace; font-size: 11px; fill: #334155; }",
        "    .section-title { font-size: 13px; font-weight: 700; fill: #1e293b; }",
        "    .arrow-label { font-size: 11px; font-weight: 600; }",
        "    .step-num { font-size: 11px; font-weight: 700; fill: #fff; }",
        "  </style>",
    ]
    for mid, color in markers:
        L.append(
            f'  <marker id="{mid}" viewBox="0 0 10 8" refX="9" refY="4"'
            ' markerWidth="8" markerHeight="6" orient="auto-start-reverse">'
        )
        L.append(f'    <path d="M0,0 L10,4 L0,8 Z" fill="{color}"/>')
        L.append("  </marker>")
    L.append("</defs>")
    return L


def card(
    L: list[str],
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body: str,
    *,
    stroke: str,
    band: str,
    title_fill: str,
) -> None:
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#fff" stroke="{stroke}"/>')
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="22" rx="8" fill="{band}"/>')
    L.append(f'<rect x="{x}" y="{y + 14}" width="{w}" height="8" fill="{band}"/>')
    cx = x + w / 2
    L.append(
        f'<text x="{cx}" y="{y + 16}" text-anchor="middle" class="node-title" fill="{title_fill}">{e(title)}</text>'
    )
    L.append(f'<text x="{cx}" y="{y + 42}" text-anchor="middle" class="node-text">{e(body)}</text>')


def step_badge(L: list[str], x: float, y: float, n: str, fill: str) -> None:
    L.append(f'<circle cx="{x}" cy="{y}" r="11" fill="{fill}"/>')
    L.append(f'<text x="{x}" y="{y + 4}" text-anchor="middle" class="step-num">{n}</text>')


def gen_ced() -> Path:
    W, H = 1120, 780
    L: list[str] = []
    L.append('<?xml version="1.0" encoding="UTF-8"?>')
    L.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"'
        ' font-family="PingFang SC,Microsoft YaHei,Helvetica Neue,Arial,sans-serif">'
    )
    L.extend(defs())
    L.append('<rect width="100%" height="100%" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>')

    L.append(f'<text x="36" y="32" class="title">{e("CED 推理过程：Prefill 半深备 KV · Decode 全深读投影")}</text>')
    L.append(
        f'<text x="36" y="52" class="subtitle">'
        f'{e("40 层 = 因果 encoder(1–20) + decoder(21–40)；decoder 全局 KV 由 H₂₀ 经层相关投影得到")}</text>'
    )

    # Architecture spine (top)
    L.append('<rect x="28" y="68" width="1064" height="118" rx="10" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.5"/>')
    L.append(f'<text x="48" y="92" class="section-title">{e("网络骨架（一次前向的层分工）")}</text>')

    # Encoder block
    L.append('<rect x="48" y="106" width="420" height="60" rx="8" fill="#fff" stroke="#60a5fa"/>')
    L.append('<rect x="48" y="106" width="420" height="22" rx="8" fill="#dbeafe"/>')
    L.append('<rect x="48" y="120" width="420" height="8" fill="#dbeafe"/>')
    L.append(f'<text x="258" y="122" text-anchor="middle" class="node-title" fill="#1d4ed8">{e("Causal Encoder · L1–L20")}</text>')
    L.append(f'<text x="258" y="148" text-anchor="middle" class="node-text">{e("本层算 attention + SWA · 末层输出 H₂₀")}</text>')

    L.append('<line x1="468" y1="136" x2="508" y2="136" stroke="#7c3aed" stroke-width="2" marker-end="url(#arrow-violet)"/>')
    L.append(f'<text x="488" y="124" text-anchor="middle" class="arrow-label" fill="#7c3aed">{e("H₂₀")}</text>')

    # Projection
    L.append('<rect x="508" y="106" width="200" height="60" rx="8" fill="#fff" stroke="#c4b5fd"/>')
    L.append('<rect x="508" y="106" width="200" height="22" rx="8" fill="#ede9fe"/>')
    L.append('<rect x="508" y="120" width="200" height="8" fill="#ede9fe"/>')
    L.append(f'<text x="608" y="122" text-anchor="middle" class="node-title" fill="#6d28d9">{e("层相关投影")}</text>')
    L.append(f'<text x="608" y="148" text-anchor="middle" class="mono">{e("Cₗ = H₂₀ Wₗ")}</text>')

    L.append('<line x1="708" y1="136" x2="748" y2="136" stroke="#ea580c" stroke-width="2" marker-end="url(#arrow-orange)"/>')

    # Decoder block
    L.append('<rect x="748" y="106" width="320" height="60" rx="8" fill="#fff" stroke="#fb923c"/>')
    L.append('<rect x="748" y="106" width="320" height="22" rx="8" fill="#ffedd5"/>')
    L.append('<rect x="748" y="120" width="320" height="8" fill="#ffedd5"/>')
    L.append(f'<text x="908" y="122" text-anchor="middle" class="node-title" fill="#9a3412">{e("Decoder · L21–L40")}</text>')
    L.append(f'<text x="908" y="148" text-anchor="middle" class="node-text">{e("读投影全局 KV · 本层仍写 SWA")}</text>')

    # Prefill panel
    L.append('<rect x="28" y="204" width="520" height="500" rx="10" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.5"/>')
    L.append('<rect x="44" y="218" width="88" height="26" rx="13" fill="#2563eb"/>')
    L.append(f'<text x="88" y="236" text-anchor="middle" class="badge-text">Prefill</text>')
    L.append(f'<text x="146" y="236" class="section-title">{e("激活 ≈ 8B · 只跑前半网络备全局 KV")}</text>')

    py = 262
    steps_prefill = [
        ("1", "输入 prompt 全序列", "工具回灌 / cache miss 触发的长 prefill"),
        ("2", "Encoder L1–L20 前向", "每层：全局分支 + 本层 SWA KV"),
        ("3", "取 H₂₀ → 投影填库", "为每个 decoder 层生成全局 KV（main / indexer）"),
        ("4", "Decoder SWA Bounded Replay", "仅重放末 n_win token，近似重建 decoder SWA"),
        ("5", "全局 KV 就绪，跳过 decoder 全深 prefill", "prompt 对 decoder 全局注意力几乎旁路"),
    ]
    for i, (num, title, body) in enumerate(steps_prefill):
        y = py + i * 78
        step_badge(L, 58, y + 28, num, "#2563eb")
        L.append(f'<rect x="78" y="{y}" width="448" height="64" rx="8" fill="#fff" stroke="#60a5fa"/>')
        L.append(f'<rect x="78" y="{y}" width="448" height="22" rx="8" fill="#dbeafe"/>')
        L.append(f'<rect x="78" y="{y + 14}" width="448" height="8" fill="#dbeafe"/>')
        L.append(
            f'<text x="302" y="{y + 16}" text-anchor="middle" class="node-title" fill="#1d4ed8">{e(title)}</text>'
        )
        L.append(f'<text x="302" y="{y + 44}" text-anchor="middle" class="node-text">{e(body)}</text>')
        if i < len(steps_prefill) - 1:
            L.append(
                f'<line x1="58" y1="{y + 64}" x2="58" y2="{y + 78}" stroke="#3b82f6"'
                ' stroke-width="1.6" marker-end="url(#arrow-blue)"/>'
            )

    # Decode panel
    L.append('<rect x="568" y="204" width="524" height="500" rx="10" fill="#fff7ed" stroke="#fdba74" stroke-width="1.5"/>')
    L.append('<rect x="584" y="218" width="88" height="26" rx="13" fill="#ea580c"/>')
    L.append(f'<text x="628" y="236" text-anchor="middle" class="badge-text">Decode</text>')
    L.append(f'<text x="686" y="236" class="section-title">{e("激活 ≈ 16B · 新 token 走全 40 层")}</text>')

    steps_decode = [
        ("1", "新生成 token 入网", "autoregressive；上下文已在全局 KV 库中"),
        ("2", "Encoder L1–L20 更新", "追加本 token 的 encoder 侧 KV / SWA"),
        ("3", "H₂₀ 再投影（或增量写）", "decoder 全局 KV 继续由 encoder 末隐状态驱动"),
        ("4", "Decoder L21–L40 读库", "全局注意力读投影 KV；本层算 Q + SWA"),
        ("5", "产出 logits / 下一 token", "相对 prefill：深度翻倍，但序列长 = 1"),
    ]
    for i, (num, title, body) in enumerate(steps_decode):
        y = py + i * 78
        step_badge(L, 598, y + 28, num, "#ea580c")
        L.append(f'<rect x="618" y="{y}" width="452" height="64" rx="8" fill="#fff" stroke="#fb923c"/>')
        L.append(f'<rect x="618" y="{y}" width="452" height="22" rx="8" fill="#ffedd5"/>')
        L.append(f'<rect x="618" y="{y + 14}" width="452" height="8" fill="#ffedd5"/>')
        L.append(
            f'<text x="844" y="{y + 16}" text-anchor="middle" class="node-title" fill="#9a3412">{e(title)}</text>'
        )
        L.append(f'<text x="844" y="{y + 44}" text-anchor="middle" class="node-text">{e(body)}</text>')
        if i < len(steps_decode) - 1:
            L.append(
                f'<line x1="598" y1="{y + 64}" x2="598" y2="{y + 78}" stroke="#ea580c"'
                ' stroke-width="1.6" marker-end="url(#arrow-orange)"/>'
            )

    # Bottom note strip
    L.append('<rect x="28" y="720" width="1064" height="42" rx="8" fill="#f5f3ff" stroke="#c4b5fd"/>')
    L.append(
        f'<text x="560" y="746" text-anchor="middle" class="node-text">'
        f'{e("官方口径：序列 ≫ n_win 时 prefill 复杂度约 O(L) → O(L/2)；agent 输入重场景专门受益")}</text>'
    )

    L.append("</svg>")
    return write_svg("ced-inference-process.svg", L)


def gen_csa2() -> Path:
    W, H = 1180, 860
    L: list[str] = []
    L.append('<?xml version="1.0" encoding="UTF-8"?>')
    L.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"'
        ' font-family="PingFang SC,Microsoft YaHei,Helvetica Neue,Arial,sans-serif">'
    )
    L.extend(defs())
    L.append('<rect width="100%" height="100%" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>')

    L.append(f'<text x="36" y="32" class="title">{e("CSA2 推理过程：三模式装配 + 解码侧候选池")}</text>')
    L.append(
        f'<text x="36" y="52" class="subtitle">'
        f'{e("每层仍算 main Q 与 SWA KV；差别只在 main KV / indexer K / Top-K 从哪来")}</text>'
    )

    # Shared core assembly
    L.append('<rect x="28" y="68" width="1124" height="100" rx="10" fill="#f0fdf4" stroke="#86efac" stroke-width="1.5"/>')
    L.append(f'<text x="48" y="92" class="section-title">{e("单层 Core Attention 装配（三种模式共用）")}</text>')

    boxes = [
        (48, "main Q", "本层 hidden 投影"),
        (268, "选中 main KV", "Top-K 稀疏条目"),
        (508, "本层 SWA KV", "窗口内精确局部"),
        (748, "Attention Out", "Q × (KV ∪ SWA)"),
        (968, "FFN / MoE", "写回残差流"),
    ]
    for i, (x, title, body) in enumerate(boxes):
        L.append(f'<rect x="{x}" y="108" width="200" height="44" rx="8" fill="#fff" stroke="#4ade80"/>')
        L.append(f'<text x="{x + 100}" y="126" text-anchor="middle" class="mono">{e(title)}</text>')
        L.append(f'<text x="{x + 100}" y="142" text-anchor="middle" class="node-text">{e(body)}</text>')
        if i < len(boxes) - 1:
            nx = boxes[i + 1][0]
            L.append(
                f'<line x1="{x + 200}" y1="130" x2="{nx}" y2="130" stroke="#0d9488"'
                ' stroke-width="1.6" marker-end="url(#arrow-teal)"/>'
            )

    # Three mode columns
    modes = [
        {
            "x": 28,
            "badge": "Full",
            "fill": "#2563eb",
            "panel": "#eff6ff",
            "stroke": "#93c5fd",
            "card_stroke": "#60a5fa",
            "band": "#dbeafe",
            "title_fill": "#1d4ed8",
            "arrow": "arrow-blue",
            "subtitle": "算全新全局状态",
            "steps": [
                ("算 main KV", "本层压缩 / 写入全局 cache"),
                ("算 indexer Q", "本层 query 打分向量"),
                ("indexer K ← main KV", "由 main KV 投影得到"),
                ("全长（或压缩序列）打分", "产出新 Top-K"),
                ("装配 Core Attention", "Q + Top-K KV + SWA"),
            ],
        },
        {
            "x": 412,
            "badge": "Reindex",
            "fill": "#7c3aed",
            "panel": "#f5f3ff",
            "stroke": "#c4b5fd",
            "card_stroke": "#a78bfa",
            "band": "#ede9fe",
            "title_fill": "#6d28d9",
            "arrow": "arrow-violet",
            "subtitle": "复用 KV · 重打分",
            "steps": [
                ("读最近 Full 的 main KV", "与对应 indexer K 一并复用"),
                ("算本层 indexer Q", "层间选择可变化"),
                ("在复用 K 上再打分", "decoder 深层限候选池，见下"),
                ("产出本层新 Top-K", "main KV 仍共享"),
                ("装配 Core Attention", "Q + 新 Top-K + SWA"),
            ],
        },
        {
            "x": 796,
            "badge": "Reuse",
            "fill": "#ea580c",
            "panel": "#fff7ed",
            "stroke": "#fdba74",
            "card_stroke": "#fb923c",
            "band": "#ffedd5",
            "title_fill": "#9a3412",
            "arrow": "arrow-orange",
            "subtitle": "KV + Top-K 全复用",
            "steps": [
                ("读最近可用 main KV", "来自上游 Full / Reindex"),
                ("直接取已有 Top-K", "本层 indexer 不参与"),
                ("本层仍算 main Q", "查询向量层内独立"),
                ("本层仍写 SWA KV", "局部窗口层内独立"),
                ("装配 Core Attention", "最少 kernel（decode ~11）"),
            ],
        },
    ]

    for m in modes:
        x = m["x"]
        L.append(
            f'<rect x="{x}" y="188" width="356" height="360" rx="10" fill="{m["panel"]}"'
            f' stroke="{m["stroke"]}" stroke-width="1.5"/>'
        )
        L.append(f'<rect x="{x + 16}" y="202" width="84" height="26" rx="13" fill="{m["fill"]}"/>')
        L.append(
            f'<text x="{x + 58}" y="220" text-anchor="middle" class="badge-text">{m["badge"]}</text>'
        )
        L.append(f'<text x="{x + 112}" y="220" class="section-title">{e(m["subtitle"])}</text>')

        for i, (title, body) in enumerate(m["steps"]):
            y = 242 + i * 56
            L.append(
                f'<rect x="{x + 16}" y="{y}" width="324" height="48" rx="8" fill="#fff"'
                f' stroke="{m["card_stroke"]}"/>'
            )
            L.append(f'<rect x="{x + 16}" y="{y}" width="324" height="20" rx="8" fill="{m["band"]}"/>')
            L.append(f'<rect x="{x + 16}" y="{y + 12}" width="324" height="8" fill="{m["band"]}"/>')
            L.append(
                f'<text x="{x + 178}" y="{y + 14}" text-anchor="middle" class="node-title"'
                f' fill="{m["title_fill"]}">{e(title)}</text>'
            )
            L.append(
                f'<text x="{x + 178}" y="{y + 36}" text-anchor="middle" class="node-text">{e(body)}</text>'
            )
            if i < len(m["steps"]) - 1:
                L.append(
                    f'<line x1="{x + 178}" y1="{y + 48}" x2="{x + 178}" y2="{y + 56}"'
                    f' stroke="{m["fill"]}" stroke-width="1.4" marker-end="url(#{m["arrow"]})"/>'
                )

    # Hierarchical indexer bottom
    L.append('<rect x="28" y="566" width="1124" height="226" rx="10" fill="#ecfeff" stroke="#67e8f9" stroke-width="1.5"/>')
    L.append('<rect x="44" y="580" width="210" height="26" rx="13" fill="#0891b2"/>')
    L.append(f'<text x="149" y="598" text-anchor="middle" class="badge-text">Hierarchical Sparse Indexer</text>')
    L.append(f'<text x="268" y="598" class="section-title">{e("仅 CED decoder · 深层 indexer 与全长解耦")}</text>')

    hier = [
        (48, "① Decoder 首个 Full", "对可见上下文打分\n取 Top-512"),
        (280, "② 块级候选", "每 8 token 一块\n块分 = 块内 max score"),
        (512, "③ 候选池", "Top-2048 块\n≈ 16,384 位置"),
        (744, "④ 后续 Reindex", "只在池内再打分\n取本层 Top-512"),
        (976, "⑤ Reuse 层", "直接用上游 Top-K\n不再扩大搜索域"),
    ]
    for i, (x, title, body) in enumerate(hier):
        lines = body.split("\n")
        L.append(f'<rect x="{x}" y="622" width="212" height="100" rx="8" fill="#fff" stroke="#22d3ee"/>')
        L.append(f'<rect x="{x}" y="622" width="212" height="24" rx="8" fill="#cffafe"/>')
        L.append(f'<rect x="{x}" y="638" width="212" height="8" fill="#cffafe"/>')
        L.append(
            f'<text x="{x + 106}" y="640" text-anchor="middle" class="node-title" fill="#0e7490">{e(title)}</text>'
        )
        L.append(f'<text x="{x + 106}" y="678" text-anchor="middle" class="node-text">{e(lines[0])}</text>')
        if len(lines) > 1:
            L.append(f'<text x="{x + 106}" y="696" text-anchor="middle" class="node-text">{e(lines[1])}</text>')
        if i < len(hier) - 1:
            L.append(
                f'<line x1="{x + 212}" y1="672" x2="{x + 280}" y2="672" stroke="#0891b2"'
                ' stroke-width="1.6" marker-end="url(#arrow-teal)"/>'
            )

    L.append('<rect x="48" y="740" width="1084" height="36" rx="8" fill="#fff" stroke="#a5f3fc"/>')
    L.append(
        f'<text x="590" y="762" text-anchor="middle" class="node-text">'
        f'{e("效果：深层 indexer 每 query 代价与上下文长度解耦（固定候选池）· 再叠 FP4 main KV → ≈ 890 B/token")}</text>'
    )

    L.append("</svg>")
    return write_svg("csa2-inference-process.svg", L)


def gen_kv_types_core_attention() -> Path:
    """How main KV / indexer / Top-K / SWA relate in one layer Core Attention."""
    W, H = 1160, 880
    V4_FIG = ROOT / "docs" / "figures" / "v4"
    L: list[str] = []
    L.append('<?xml version="1.0" encoding="UTF-8"?>')
    L.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"'
        ' font-family="PingFang SC,Microsoft YaHei,Helvetica Neue,Arial,sans-serif">'
    )
    L.extend(defs())
    L.append('<rect width="100%" height="100%" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>')

    L.append(f'<text x="36" y="32" class="title">{e("异构 KV：单层 Core Attention 里谁被读、谁只负责选块")}</text>')
    L.append(
        f'<text x="36" y="52" class="subtitle">'
        f'{e("同一 main Q 并行读两路记忆 — 全局压缩稀疏（main KV）+ 局部精确（SWA）；V4.1 全局侧为 CSA2 main KV")}</text>'
    )

    # --- Sequence axis ---
    L.append('<rect x="28" y="68" width="1104" height="118" rx="10" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.5"/>')
    L.append(f'<text x="48" y="90" class="section-title">{e("沿时间轴：历史 token 如何变成两类可被读的对象")}</text>')

    ax_y, ax_h = 102, 36
    L.append(f'<rect x="48" y="{ax_y}" width="620" height="{ax_h}" rx="6" fill="#dbeafe" stroke="#60a5fa"/>')
    L.append(
        f'<text x="358" y="{ax_y + 22}" text-anchor="middle" class="node-text">'
        f'{e("远距历史 → 压成 main KV entry（V4：CSA 4:1 / HCA 128:1 · V4.1：CSA2 + FP4）")}</text>'
    )
    L.append(f'<rect x="680" y="{ax_y}" width="300" height="{ax_h}" rx="6" fill="#ffedd5" stroke="#fb923c"/>')
    L.append(
        f'<text x="830" y="{ax_y + 22}" text-anchor="middle" class="node-text">'
        f'{e("最近 n_win token → 每 token 一条 SWA K/V（不压缩）")}</text>'
    )
    L.append(f'<rect x="992" y="{ax_y}" width="120" height="{ax_h}" rx="6" fill="#e2e8f0" stroke="#94a3b8"/>')
    L.append(
        f'<text x="1052" y="{ax_y + 22}" text-anchor="middle" class="node-text">{e("当前 token")}</text>'
    )
    L.append(
        f'<text x="580" y="168" text-anchor="middle" class="node-text">'
        f'{e("Tail：未凑满压缩块的尾巴先暂存，凑满后再进 main KV（V4 专名；V4.1 同理逻辑）")}</text>'
    )

    # --- Core attention flow (tall enough for merge box) ---
    L.append('<rect x="28" y="198" width="1104" height="390" rx="10" fill="#fff" stroke="#94a3b8" stroke-width="1.5"/>')
    L.append(f'<text x="48" y="222" class="section-title">{e("单层推理：main Q 分两路，最后在 Core Attention 合并")}</text>')

    # main Q source
    L.append('<rect x="480" y="238" width="200" height="48" rx="8" fill="#f5f3ff" stroke="#a78bfa"/>')
    L.append('<rect x="480" y="238" width="200" height="20" rx="8" fill="#ede9fe"/>')
    L.append('<rect x="480" y="250" width="200" height="8" fill="#ede9fe"/>')
    L.append(
        f'<text x="580" y="252" text-anchor="middle" class="node-title" fill="#6d28d9">{e("本层 hidden")}</text>'
    )
    L.append(f'<text x="580" y="274" text-anchor="middle" class="mono">{e("→ main Q")}</text>')
    L.append(
        '<line x1="580" y1="286" x2="580" y2="304" stroke="#7c3aed" stroke-width="2" marker-end="url(#arrow-violet)"/>'
    )
    L.append(
        '<line x1="580" y1="304" x2="320" y2="304" stroke="#3b82f6" stroke-width="1.8"/>'
    )
    L.append(
        '<line x1="580" y1="304" x2="840" y2="304" stroke="#ea580c" stroke-width="1.8"/>'
    )
    L.append(
        '<line x1="320" y1="304" x2="320" y2="318" stroke="#3b82f6" stroke-width="1.8" marker-end="url(#arrow-blue)"/>'
    )
    L.append(
        '<line x1="840" y1="304" x2="840" y2="318" stroke="#ea580c" stroke-width="1.8" marker-end="url(#arrow-orange)"/>'
    )

    # Global branch panel
    L.append('<rect x="48" y="318" width="520" height="180" rx="10" fill="#eff6ff" stroke="#93c5fd" stroke-width="1.5"/>')
    L.append(f'<text x="68" y="340" class="section-title" fill="#1d4ed8">{e("全局 · 稀疏（main 分支）")}</text>')

    card = [
        (68, 352, 220, 52, "main KV", "压缩后的全局 K/V 条目", "#dbeafe", "#1d4ed8"),
        (308, 352, 220, 52, "indexer K", "由 main KV 投影（CSA2）", "#dbeafe", "#1d4ed8"),
        (68, 414, 220, 52, "indexer Q", "本层打分 query", "#dbeafe", "#1d4ed8"),
        (308, 414, 220, 52, "Top-K 索引", "选哪些 main entry 参与", "#dbeafe", "#1d4ed8"),
    ]
    for x, y, w, h, title, body, band, tf in card:
        L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#fff" stroke="#60a5fa"/>')
        L.append(f'<rect x="{x}" y="{y}" width="{w}" height="20" rx="8" fill="{band}"/>')
        L.append(f'<rect x="{x}" y="{y + 12}" width="{w}" height="8" fill="{band}"/>')
        L.append(f'<text x="{x + w / 2}" y="{y + 14}" text-anchor="middle" class="node-title" fill="{tf}">{e(title)}</text>')
        L.append(f'<text x="{x + w / 2}" y="{y + 38}" text-anchor="middle" class="node-text">{e(body)}</text>')

    L.append(
        '<line x1="288" y1="466" x2="288" y2="482" stroke="#3b82f6" stroke-width="1.4" marker-end="url(#arrow-blue)"/>'
    )
    L.append('<rect x="188" y="482" width="200" height="28" rx="8" fill="#2563eb"/>')
    L.append(
        f'<text x="288" y="500" text-anchor="middle" class="badge-text">{e("main Q × 选中 main KV")}</text>'
    )

    # SWA branch panel
    L.append('<rect x="592" y="318" width="520" height="180" rx="10" fill="#fff7ed" stroke="#fdba74" stroke-width="1.5"/>')
    L.append(f'<text x="612" y="340" class="section-title" fill="#9a3412">{e("局部 · SWA（单独一路）")}</text>')

    L.append('<rect x="632" y="356" width="440" height="70" rx="8" fill="#fff" stroke="#fb923c"/>')
    L.append('<rect x="632" y="356" width="440" height="22" rx="8" fill="#ffedd5"/>')
    L.append('<rect x="632" y="370" width="440" height="8" fill="#ffedd5"/>')
    L.append(
        f'<text x="852" y="372" text-anchor="middle" class="node-title" fill="#9a3412">{e("SWA KV")}</text>'
    )
    L.append(
        f'<text x="852" y="396" text-anchor="middle" class="node-text">'
        f'{e("本层、窗口内、每 token 精确 K/V · 与 main 分池（State）")}</text>'
    )
    L.append(
        f'<text x="852" y="414" text-anchor="middle" class="node-text">'
        f'{e("不参与 Top-K；也不压成 4:1 / 128:1 entry")}</text>'
    )
    L.append('<rect x="692" y="440" width="320" height="28" rx="8" fill="#ea580c"/>')
    L.append(
        f'<text x="852" y="458" text-anchor="middle" class="badge-text">{e("main Q × SWA KV（dense 局部）")}</text>'
    )

    # Merge (inside middle panel)
    L.append(
        '<line x1="288" y1="510" x2="580" y2="530" stroke="#64748b" stroke-width="2" marker-end="url(#arrow-slate)"/>'
    )
    L.append(
        '<line x1="852" y1="468" x2="580" y2="530" stroke="#64748b" stroke-width="2" marker-end="url(#arrow-slate)"/>'
    )
    L.append('<rect x="460" y="530" width="240" height="44" rx="10" fill="#ecfdf5" stroke="#34d399" stroke-width="2"/>')
    L.append(
        f'<text x="580" y="550" text-anchor="middle" class="node-title" fill="#047857">{e("Core Attention 输出")}</text>'
    )
    L.append(
        f'<text x="580" y="566" text-anchor="middle" class="node-text">{e("两路加权和 → 本层 attention 结果")}</text>'
    )

    # --- Role cards ---
    L.append('<rect x="28" y="604" width="1104" height="118" rx="10" fill="#f8fafc" stroke="#e2e8f0"/>')
    L.append(f'<text x="48" y="628" class="section-title">{e("各对象职责（易混点）")}</text>')

    roles = [
        (48, "main KV", "被 attend 的全局记忆", "压缩 · 可 FP4 · 长驻 HBM/SSD"),
        (248, "indexer K", "打分用的键", "CSA2：由 main KV 投影"),
        (448, "Top-K", "稀疏索引", "告诉 Q 读哪几条 main entry"),
        (648, "SWA KV", "被 attend 的局部记忆", "精确 · 短 TTL · V4.1 多不持久化 SSD"),
        (848, "indexer Q", "本层打分 query", "Full/Reindex 才算；Reuse 可跳过"),
    ]
    for x, title, line1, line2 in roles:
        L.append(f'<rect x="{x}" y="640" width="188" height="68" rx="8" fill="#fff" stroke="#cbd5e1"/>')
        L.append(f'<text x="{x + 94}" y="660" text-anchor="middle" class="mono">{e(title)}</text>')
        L.append(f'<text x="{x + 94}" y="678" text-anchor="middle" class="node-text">{e(line1)}</text>')
        L.append(f'<text x="{x + 94}" y="694" text-anchor="middle" class="node-text">{e(line2)}</text>')

    # Why SWA strip
    L.append('<rect x="28" y="738" width="1104" height="118" rx="10" fill="#fff7ed" stroke="#fdba74" stroke-width="1.5"/>')
    L.append(f'<text x="48" y="762" class="section-title" fill="#9a3412">{e("为何单独要 SWA？")}</text>')
    L.append(
        f'<text x="48" y="786" class="node-text">'
        f'{e("① 块压缩会丢 token 级细节，最近邻域对语法 / 指代 / 工具格式极敏感")}</text>'
    )
    L.append(
        f'<text x="48" y="806" class="node-text">'
        f'{e("② 远距用 main KV + Top-K 省算力与显存；近距用 SWA 保精度 — 互补，不能互相替代")}</text>'
    )
    L.append(
        f'<text x="48" y="826" class="node-text">'
        f'{e("③ 生命周期不同：全局 main 可 prefix 共享 + 落盘；SWA 短窗口、层内 state，V4.1 用 Bounded Replay 近似重建")}</text>'
    )

    L.append("</svg>")
    return write_svg(
        "kv-types-core-attention.svg",
        L,
        extra_dirs=[V4_FIG],
    )


def main() -> None:
    p1 = gen_ced()
    p2 = gen_csa2()
    p3 = gen_kv_types_core_attention()
    print(p1)
    print(p2)
    print(p3)


if __name__ == "__main__":
    main()
