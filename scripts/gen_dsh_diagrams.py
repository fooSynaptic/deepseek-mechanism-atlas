#!/usr/bin/env python3
"""Generate DeepSeek Harness architecture SVGs (CJK as hex entities)."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "diagrams"


def e(s: str) -> str:
    return "".join(f"&#x{ord(c):X};" if ord(c) > 127 else c for c in s)


def write_svg(name: str, lines: list[str]) -> Path:
    path = OUT_DIR / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def gen_architecture() -> Path:
    W, H = 1100, 740
    L: list[str] = []
    L.append('<?xml version="1.0" encoding="UTF-8"?>')
    L.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"'
        ' font-family="PingFang SC,Microsoft YaHei,Helvetica Neue,Arial,sans-serif">'
    )
    L.append("<defs>")
    L.append("  <style>")
    L.append("    .title { font-size: 20px; font-weight: 700; fill: #1a1a2e; }")
    L.append("    .subtitle { font-size: 12px; fill: #64748b; }")
    L.append("    .badge-text { font-size: 11px; font-weight: 600; fill: #fff; }")
    L.append("    .node-title { font-size: 13px; font-weight: 600; fill: #1e293b; }")
    L.append("    .node-text { font-size: 11px; fill: #475569; }")
    L.append("    .mono { font-family: Consolas,Menlo,monospace; font-size: 11px; fill: #334155; }")
    L.append("    .section-title { font-size: 13px; font-weight: 700; fill: #1e293b; }")
    L.append("    .arrow-label { font-size: 11px; font-weight: 600; fill: #3b82f6; }")
    L.append("  </style>")
    for mid, color in [
        ("arrow-blue", "#3b82f6"),
        ("arrow-violet", "#7c3aed"),
        ("arrow-slate", "#64748b"),
    ]:
        L.append(
            f'  <marker id="{mid}" viewBox="0 0 10 8" refX="9" refY="4"'
            ' markerWidth="8" markerHeight="6" orient="auto-start-reverse">'
        )
        L.append(f'    <path d="M0,0 L10,4 L0,8 Z" fill="{color}"/>')
        L.append("  </marker>")
    L.append("</defs>")
    L.append('<rect width="100%" height="100%" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>')

    # Title
    L.append(f'<text x="40" y="36" class="title">{e("DeepSeek Harness · 整体设计架构")}</text>')
    L.append(
        f'<text x="40" y="58" class="subtitle">'
        f'{e("Everything is a plugin · Cordis ctx · Profile / Bundle / Patch")}</text>'
    )

    # Layer 1: Surfaces / Profiles
    L.append('<rect x="28" y="78" width="1044" height="108" rx="10" fill="#eef2ff" stroke="#a5b4fc" stroke-width="1.5"/>')
    L.append('<rect x="44" y="90" width="92" height="26" rx="13" fill="#4f46e5"/>')
    L.append(f'<text x="90" y="108" text-anchor="middle" class="badge-text">{e("入口层")}</text>')
    L.append(f'<text x="150" y="108" class="section-title">{e("Profiles / Surfaces")}</text>')

    profiles = [
        (48, "web", e("浏览器 UI")),
        (248, "headless", e("一次性 runner")),
        (448, "sdk / sdk-min", e("JSON-RPC")),
        (648, "acp", e("自动化服务")),
        (848, "desktop", e("Electron")),
    ]
    for x, name, desc in profiles:
        L.append(f'<rect x="{x}" y="128" width="180" height="44" rx="8" fill="#fff" stroke="#818cf8"/>')
        L.append(f'<rect x="{x}" y="128" width="180" height="22" rx="8" fill="#e0e7ff"/>')
        L.append(f'<rect x="{x}" y="142" width="180" height="8" fill="#e0e7ff"/>')
        L.append(f'<text x="{x + 90}" y="144" text-anchor="middle" class="mono">{name}</text>')
        L.append(f'<text x="{x + 90}" y="162" text-anchor="middle" class="node-text">{desc}</text>')

    # Arrow down
    L.append('<line x1="550" y1="186" x2="550" y2="214" stroke="#7c3aed" stroke-width="2" marker-end="url(#arrow-violet)"/>')
    L.append(f'<text x="568" y="206" class="arrow-label">compose</text>')

    # Layer 2: Cordis core
    L.append('<rect x="28" y="218" width="1044" height="250" rx="10" fill="#ecfdf5" stroke="#6ee7b7" stroke-width="1.5"/>')
    L.append('<rect x="44" y="230" width="118" height="26" rx="13" fill="#059669"/>')
    L.append(f'<text x="103" y="248" text-anchor="middle" class="badge-text">{e("Cordis 内核")}</text>')
    L.append(
        f'<text x="180" y="248" class="section-title">'
        f'{e("共享 ctx · services / events / reversible effects")}</text>'
    )

    cores = [
        (48, 270, "#dbeafe", "#93c5fd", "#1d4ed8", "ctx.llm", e("模型适配")),
        (220, 270, "#fce7f3", "#f9a8d4", "#be185d", "ctx.tools", e("工具注册 / 管线")),
        (392, 270, "#fef3c7", "#fcd34d", "#b45309", "ctx.agentLoop", e("默认 driver")),
        (564, 270, "#e0e7ff", "#a5b4fc", "#4338ca", "ctx.agents", e("Agent 注册表")),
        (736, 270, "#dcfce7", "#86efac", "#166534", "ctx.sessions", e("SessionEvent log")),
        (908, 270, "#ffedd5", "#fdba74", "#c2410c", "ctx.systemPrompt", e("Prompt 装配")),
        (136, 360, "#ede9fe", "#c4b5fd", "#6d28d9", "ctx.sandbox", e("执行隔离")),
        (348, 360, "#e0f2fe", "#7dd3fc", "#0369a1", "codeRuntime", e("Code Mode 沙箱")),
        (560, 360, "#f1f5f9", "#cbd5e1", "#334155", "webhook / jobs", e("外部触发 / 后台")),
        (772, 360, "#fae8ff", "#e879f9", "#a21caf", "UI / ACP", e("表面渲染")),
    ]
    for x, y, band, stroke, title_fill, name, desc in cores:
        L.append(f'<rect x="{x}" y="{y}" width="156" height="70" rx="8" fill="#fff" stroke="{stroke}"/>')
        L.append(f'<rect x="{x}" y="{y}" width="156" height="24" rx="8" fill="{band}"/>')
        L.append(f'<rect x="{x}" y="{y + 16}" width="156" height="8" fill="{band}"/>')
        L.append(
            f'<text x="{x + 78}" y="{y + 18}" text-anchor="middle" class="mono" fill="{title_fill}">{name}</text>'
        )
        L.append(f'<text x="{x + 78}" y="{y + 48}" text-anchor="middle" class="node-text">{desc}</text>')

    L.append(
        f'<text x="550" y="452" text-anchor="middle" class="node-text">'
        f'{e("卸载插件 → 注册自动回滚；扩展 = 再挂一个插件")}</text>'
    )

    # Arrow down
    L.append('<line x1="550" y1="468" x2="550" y2="496" stroke="#3b82f6" stroke-width="2" marker-end="url(#arrow-blue)"/>')
    L.append(f'<text x="568" y="488" class="arrow-label">boot layers</text>')

    # Layer 3: Composition
    L.append('<rect x="28" y="500" width="680" height="120" rx="10" fill="#fff7ed" stroke="#fdba74" stroke-width="1.5"/>')
    L.append('<rect x="44" y="512" width="118" height="26" rx="13" fill="#ea580c"/>')
    L.append(f'<text x="103" y="530" text-anchor="middle" class="badge-text">{e("组合层")}</text>')
    L.append(f'<text x="180" y="530" class="section-title">Bundle → profile patch → home → --patch</text>')

    steps = [
        (48, "1. bundles", "dsh-base + app"),
        (210, "2. profile patch", "cordis.patch.yml"),
        (372, "3. home patch", e("用户级覆盖")),
        (534, "4. --patch", e("CLI overlay")),
    ]
    for i, (x, title, sub) in enumerate(steps):
        L.append(f'<rect x="{x}" y="552" width="148" height="52" rx="8" fill="#fff" stroke="#fb923c"/>')
        L.append(f'<text x="{x + 74}" y="574" text-anchor="middle" class="node-title">{title}</text>')
        L.append(f'<text x="{x + 74}" y="592" text-anchor="middle" class="node-text">{sub}</text>')
        if i < len(steps) - 1:
            L.append(
                f'<line x1="{x + 148}" y1="578" x2="{x + 162}" y2="578"'
                ' stroke="#fb923c" stroke-width="1.8" marker-end="url(#arrow-blue)"/>'
            )

    # Modes panel
    L.append('<rect x="728" y="500" width="344" height="120" rx="10" fill="#fdf4ff" stroke="#e879f9" stroke-width="1.5"/>')
    L.append('<rect x="744" y="512" width="92" height="26" rx="13" fill="#c026d3"/>')
    L.append(f'<text x="790" y="530" text-anchor="middle" class="badge-text">Modes</text>')
    modes = [
        (748, "Standard", e("全工具 agent")),
        (908, "Code", e("TS 编排工具")),
        (748, "Minimal", e("两工具基准")),
        (908, "Creator", e("拼 preset")),
    ]
    coords = [(748, 550), (908, 550), (748, 586), (908, 586)]
    labels = [
        ("Standard", e("全工具")),
        ("Code", e("TS 编排")),
        ("Minimal", e("两工具测模型")),
        ("Creator", e("拼 preset")),
    ]
    for (x, y), (name, desc) in zip(coords, labels):
        L.append(f'<rect x="{x}" y="{y}" width="144" height="28" rx="6" fill="#fff" stroke="#f0abfc"/>')
        L.append(f'<text x="{x + 10}" y="{y + 18}" class="mono">{name}</text>')
        L.append(f'<text x="{x + 78}" y="{y + 18}" class="node-text">{desc}</text>')

    # Footer note line
    L.append(
        f'<text x="550" y="652" text-anchor="middle" class="subtitle">'
        f'{e("事实源：append-only SessionEvent · 工具路径 pre-execute → execute → post-execute → result")}</text>'
    )
    L.append(
        f'<text x="550" y="674" text-anchor="middle" class="subtitle">'
        f'dump-config: dsh --profile web --dump-config</text>'
    )

    # Tighten: H was 740, content ends ~680
    L.append("</svg>")
    # Fix viewBox height - regenerate with H=690
    text = "\n".join(L).replace(f'viewBox="0 0 {W} {H}"', f'viewBox="0 0 {W} 690"')
    path = OUT_DIR / "dsh-architecture.svg"
    path.write_text(text + "\n", encoding="utf-8")
    return path


def gen_boost() -> Path:
    W, H = 1120, 720
    L: list[str] = []
    L.append('<?xml version="1.0" encoding="UTF-8"?>')
    L.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"'
        ' font-family="PingFang SC,Microsoft YaHei,Helvetica Neue,Arial,sans-serif">'
    )
    L.append("<defs>")
    L.append("  <style>")
    L.append("    .title { font-size: 20px; font-weight: 700; fill: #1a1a2e; }")
    L.append("    .subtitle { font-size: 12px; fill: #64748b; }")
    L.append("    .badge-text { font-size: 11px; font-weight: 600; fill: #fff; }")
    L.append("    .node-title { font-size: 13px; font-weight: 600; fill: #1e293b; }")
    L.append("    .node-text { font-size: 11px; fill: #475569; }")
    L.append("    .mono { font-family: Consolas,Menlo,monospace; font-size: 11px; fill: #334155; }")
    L.append("    .section-title { font-size: 13px; font-weight: 700; fill: #1e293b; }")
    L.append("    .arrow-label { font-size: 10.5px; font-weight: 600; fill: #2563eb; }")
    L.append("    .boost-label { font-size: 10.5px; font-weight: 600; fill: #b45309; }")
    L.append("  </style>")
    for mid, color in [
        ("arrow-blue", "#3b82f6"),
        ("arrow-green", "#16a34a"),
        ("arrow-amber", "#d97706"),
    ]:
        L.append(
            f'  <marker id="{mid}" viewBox="0 0 10 8" refX="9" refY="4"'
            ' markerWidth="8" markerHeight="6" orient="auto-start-reverse">'
        )
        L.append(f'    <path d="M0,0 L10,4 L0,8 Z" fill="{color}"/>')
        L.append("  </marker>")
    L.append("</defs>")
    L.append('<rect width="100%" height="100%" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>')

    L.append(f'<text x="40" y="34" class="title">{e("Agent 运行时 ↔ 模型交互 · 能力增益")}</text>')
    L.append(
        f'<text x="40" y="54" class="subtitle">'
        f'{e("左：dsh 循环 · 中：模型请求面 · 右：模型/推理特性如何抬升 harness 表现")}</text>'
    )

    # Shared vertical band for three columns
    col_top = 70
    col_h = 560

    # Left: Agent runtime — badge row then cards (no overlap)
    L.append(
        f'<rect x="24" y="{col_top}" width="330" height="{col_h}" rx="10"'
        ' fill="#eff6ff" stroke="#93c5fd" stroke-width="1.5"/>'
    )
    L.append('<rect x="40" y="82" width="128" height="26" rx="13" fill="#2563eb"/>')
    L.append(f'<text x="104" y="100" text-anchor="middle" class="badge-text">{e("Agent 运行时")}</text>')
    L.append(f'<text x="184" y="100" class="section-title">dsh</text>')

    left_cards = [
        e("用户 / 外部触发"),
        e("Turn / Step 循环"),
        e("工具管线"),
        "Code Mode",
        e("SessionEvent log"),
        e("Resume / Fork / Replay"),
    ]
    left_subs = [
        "inbox · webhook · UI",
        "pre-step → request → stream",
        "pre → execute → post → result",
        e("TS 程序 · tools.* 绑定"),
        e("模型可见 = 已入账"),
        e("同一事件流投影"),
    ]
    # First card below badge (badge ends ~108); card h=56, gap=14 → step 70
    left_y0, card_h, step = 118, 56, 70
    for i, (title, sub) in enumerate(zip(left_cards, left_subs)):
        y = left_y0 + i * step
        L.append(f'<rect x="40" y="{y}" width="298" height="{card_h}" rx="8" fill="#fff" stroke="#60a5fa"/>')
        L.append(f'<rect x="40" y="{y}" width="298" height="22" rx="8" fill="#dbeafe"/>')
        L.append(f'<rect x="40" y="{y + 14}" width="298" height="8" fill="#dbeafe"/>')
        L.append(
            f'<text x="189" y="{y + 16}" text-anchor="middle" class="node-title" fill="#1d4ed8">{title}</text>'
        )
        L.append(f'<text x="189" y="{y + 42}" text-anchor="middle" class="node-text">{sub}</text>')
        if i < len(left_cards) - 1:
            L.append(
                f'<line x1="189" y1="{y + card_h}" x2="189" y2="{y + step}"'
                ' stroke="#3b82f6" stroke-width="1.6" marker-end="url(#arrow-blue)"/>'
            )

    # Center: Model face — aligned mid-band
    L.append(
        f'<rect x="390" y="{col_top + 40}" width="300" height="480" rx="10"'
        ' fill="#f0fdf4" stroke="#86efac" stroke-width="1.5"/>'
    )
    L.append('<rect x="406" y="122" width="118" height="26" rx="13" fill="#16a34a"/>')
    L.append(f'<text x="465" y="140" text-anchor="middle" class="badge-text">{e("模型请求面")}</text>')

    mid = [
        ("ctx.llm adapter", e("消息 / 流式词汇表")),
        ("Prompt + tool schema", e("systemPrompt 装配")),
        (e("冻结请求历史"), "deriveMessages()"),
    ]
    mid_y0 = 168
    for i, (title, sub) in enumerate(mid):
        y = mid_y0 + i * 90
        L.append(f'<rect x="410" y="{y}" width="260" height="72" rx="8" fill="#fff" stroke="#4ade80"/>')
        L.append(f'<rect x="410" y="{y}" width="260" height="24" rx="8" fill="#dcfce7"/>')
        L.append(f'<rect x="410" y="{y + 16}" width="260" height="8" fill="#dcfce7"/>')
        L.append(
            f'<text x="540" y="{y + 18}" text-anchor="middle" class="node-title" fill="#166534">{title}</text>'
        )
        L.append(f'<text x="540" y="{y + 50}" text-anchor="middle" class="node-text">{sub}</text>')

    # Gutters between columns for arrow labels (clear of cards)
    # Left panel right edge 354; center left 390 → gap 354–390
    L.append(
        '<path d="M354,188 L390,204" stroke="#3b82f6" stroke-width="1.8"'
        ' fill="none" marker-end="url(#arrow-blue)"/>'
    )
    L.append(f'<text x="360" y="178" class="arrow-label">request</text>')
    L.append(
        '<path d="M390,280 L354,300" stroke="#16a34a" stroke-width="1.8"'
        ' fill="none" marker-end="url(#arrow-green)"/>'
    )
    L.append(f'<text x="358" y="268" class="arrow-label">tokens</text>')
    L.append(
        '<path d="M354,400 L390,380" stroke="#3b82f6" stroke-width="1.8"'
        ' fill="none" marker-end="url(#arrow-blue)"/>'
    )
    L.append(f'<text x="352" y="418" class="arrow-label">tool calls</text>')

    # Right: Boosters — badge then 5 cards inside panel
    L.append(
        f'<rect x="726" y="{col_top}" width="370" height="{col_h}" rx="10"'
        ' fill="#fffbeb" stroke="#fcd34d" stroke-width="1.5"/>'
    )
    L.append('<rect x="742" y="82" width="150" height="26" rx="13" fill="#d97706"/>')
    L.append(f'<text x="817" y="100" text-anchor="middle" class="badge-text">{e("模型侧增益轴")}</text>')

    boosts = [
        (e("工具 / Agent 后训练"), "V3.1+ Tool Use · SWE", e("更会调 tools / skills")),
        (e("长上下文效率"), "DSA · CSA/HCA · 1M", e("更长 tool/trace 扛得住")),
        (e("Decode 加速"), "DSpark · MTP", e("多轮 loop 更跟手")),
        (e("KV / 显存层级"), "HiSparse · ESS · Prefix", e("同卡撑更长会话")),
        (e("可验证推理"), "R1 · RLVR", e("短程硬任务更稳")),
    ]
    # badge ends ~108; card h=72, gap=16 → step 88; 5×88=440 → last ends 118+440=558 < 630
    boost_y0, boost_h, boost_step = 118, 72, 88
    for i, (title, tags, effect) in enumerate(boosts):
        y = boost_y0 + i * boost_step
        L.append(f'<rect x="742" y="{y}" width="338" height="{boost_h}" rx="8" fill="#fff" stroke="#fbbf24"/>')
        L.append(f'<rect x="742" y="{y}" width="338" height="24" rx="8" fill="#fef3c7"/>')
        L.append(f'<rect x="742" y="{y + 16}" width="338" height="8" fill="#fef3c7"/>')
        L.append(
            f'<text x="911" y="{y + 18}" text-anchor="middle" class="node-title" fill="#92400e">{title}</text>'
        )
        L.append(f'<text x="911" y="{y + 44}" text-anchor="middle" class="mono">{tags}</text>')
        L.append(f'<text x="911" y="{y + 62}" text-anchor="middle" class="node-text">{effect}</text>')

    # Center right edge 690; right panel left 726 → gap for boost arrows
    L.append(
        '<path d="M726,154 L700,220" stroke="#d97706" stroke-width="1.5"'
        ' stroke-dasharray="5,3" fill="none" marker-end="url(#arrow-amber)"/>'
    )
    L.append(f'<text x="698" y="178" class="boost-label">boost</text>')
    L.append(
        '<path d="M726,330 L700,300" stroke="#d97706" stroke-width="1.5"'
        ' stroke-dasharray="5,3" fill="none" marker-end="url(#arrow-amber)"/>'
    )
    L.append(
        '<path d="M726,418 L700,360" stroke="#d97706" stroke-width="1.5"'
        ' stroke-dasharray="5,3" fill="none" marker-end="url(#arrow-amber)"/>'
    )

    # Footer below columns (col ends 630)
    L.append(
        f'<text x="560" y="660" text-anchor="middle" class="subtitle">'
        f'{e("增益轴与 harness 正交可叠加：运行时编排不变，吞吐 / 上下文 / 工具能力随模型与 infer 栈提升")}</text>'
    )
    L.append(
        f'<text x="560" y="682" text-anchor="middle" class="subtitle">'
        f'{e("对照：Engram / MoE 路由改变权重稀疏形态；Harness 走 LLM adapter 调用已部署模型")}</text>'
    )

    L.append("</svg>")
    path = OUT_DIR / "dsh-agent-model-boost.svg"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a = gen_architecture()
    b = gen_boost()
    print(a)
    print(b)


if __name__ == "__main__":
    main()
