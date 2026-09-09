#!/usr/bin/env python3
"""Render DeepSeekMoE vs GShard architecture schematic (English labels)."""

from __future__ import annotations

from pathlib import Path

OUT = Path(__file__).resolve().parent


def t(s: str) -> str:
    raw = (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
    return "".join(f"&#x{ord(c):X};" if ord(c) > 127 else c for c in raw)


lines: list[str] = []


def L(s: str) -> None:
    lines.append(s)


def write(name: str) -> None:
    dest = OUT / name
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", dest)


def fig() -> None:
    W, H = 1100, 640
    L('<?xml version="1.0" encoding="UTF-8"?>')
    L(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"')
    L('     font-family="Helvetica Neue,Arial,sans-serif">')
    L("<defs>")
    L("  <style>")
    L("    .title { font-size: 20px; font-weight: 700; fill: #1a1a2e; }")
    L("    .subtitle { font-size: 12px; fill: #64748b; }")
    L("    .badge-text { font-size: 11px; font-weight: 600; fill: #fff; }")
    L("    .node-title { font-size: 13px; font-weight: 700; fill: #1e293b; }")
    L("    .node-text { font-size: 11.5px; fill: #475569; }")
    L("    .note { font-size: 11px; fill: #475569; }")
    L("    .inv { font-size: 12px; font-weight: 600; fill: #1d4ed8; }")
    L("  </style>")
    L('  <marker id="arrow-gray" viewBox="0 0 10 8" refX="9" refY="4"')
    L('          markerWidth="8" markerHeight="6" orient="auto-start-reverse">')
    L('    <path d="M0,0 L10,4 L0,8 Z" fill="#64748b"/>')
    L("  </marker>")
    L("</defs>")
    L('<rect width="100%" height="100%" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>')
    L(f'<text x="40" y="38" class="title">{t("Paper 2B validation: GShard vs DeepSeekMoE")}</text>')
    L(
        f'<text x="40" y="58" class="subtitle">{t("Same total expert params (16x FFN) and same activated params (2x FFN). Layout is the only variable.")}</text>'
    )

    # Invariant bar
    L('<rect x="40" y="74" width="1020" height="44" rx="10" fill="#eff6ff" stroke="#93c5fd"/>')
    L(
        f'<text x="550" y="101" text-anchor="middle" class="inv">{t("Invariant: 9L / d=1280 / 10 heads  ·  16x total experts  ·  2x activated FFN  ·  seq 2048  ·  batch 4M tokens")}</text>'
    )

    def panel(x, y, w, h, badge_fill, badge, title):
        L(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="#fff" stroke="#e2e8f0"/>')
        L(f'<rect x="{x+16}" y="{y+14}" width="220" height="24" rx="12" fill="{badge_fill}"/>')
        L(
            f'<text x="{x+126}" y="{y+30}" text-anchor="middle" class="badge-text">{t(badge)}</text>'
        )
        L(f'<text x="{x+16}" y="{y+62}" class="node-title">{t(title)}</text>')

    # GShard panel
    panel(40, 138, 500, 420, "#334155", "Arm A  ·  GShard", "16 full-size experts, Top-2")
    L('<rect x="64" y="214" width="452" height="26" rx="8" fill="#dbeafe"/>')
    L('<rect x="64" y="232" width="452" height="8" fill="#dbeafe"/>')
    L(f'<text x="290" y="233" text-anchor="middle" class="node-title" fill="#1d4ed8">{t("Router (softmax)")}</text>')
    L(f'<text x="290" y="258" text-anchor="middle" class="node-text">{t("select 2 of 16")}</text>')

    # 16 expert boxes in 2 rows
    for i in range(16):
        r, c = divmod(i, 8)
        x = 72 + c * 54
        y = 278 + r * 70
        fill = "#fef3c7" if i < 2 else "#f1f5f9"
        stroke = "#f59e0b" if i < 2 else "#cbd5e1"
        L(f'<rect x="{x}" y="{y}" width="48" height="52" rx="6" fill="{fill}" stroke="{stroke}"/>')
        L(
            f'<text x="{x+24}" y="{y+22}" text-anchor="middle" class="node-text">{t(f"E{i}")}</text>'
        )
        L(
            f'<text x="{x+24}" y="{y+40}" text-anchor="middle" class="node-text">{t("1x")}</text>'
        )
    L(
        f'<text x="290" y="430" text-anchor="middle" class="node-text">{t("Yellow = activated this token. Each expert = 1x dense FFN.")}</text>'
    )
    L(
        f'<text x="290" y="508" text-anchor="middle" class="node-title">{t("C(16,2) = 120 combinations")}</text>'
    )
    L(
        f'<text x="290" y="530" text-anchor="middle" class="note">{t("Paper: same total / activated params as DeepSeekMoE; worse Pile and downstream.")}</text>'
    )

    # DeepSeekMoE panel
    panel(560, 138, 500, 420, "#1d4ed8", "Arm B  ·  DeepSeekMoE", "1 shared + 63 routed quarter-experts")
    L('<rect x="584" y="214" width="452" height="26" rx="8" fill="#dcfce7"/>')
    L('<rect x="584" y="232" width="452" height="8" fill="#dcfce7"/>')
    L(
        f'<text x="810" y="233" text-anchor="middle" class="node-title" fill="#15803d">{t("Shared expert  (always on)")}</text>'
    )
    L('<rect x="760" y="258" width="100" height="40" rx="6" fill="#bbf7d0" stroke="#22c55e"/>')
    L(f'<text x="810" y="283" text-anchor="middle" class="node-text">{t("S0  ·  0.25x")}</text>')

    L('<rect x="584" y="312" width="452" height="26" rx="8" fill="#dbeafe"/>')
    L('<rect x="584" y="330" width="452" height="8" fill="#dbeafe"/>')
    L(
        f'<text x="810" y="331" text-anchor="middle" class="node-title" fill="#1d4ed8">{t("Router  ·  Top-7 of 63 routed")}</text>'
    )

    # 14 mini boxes to stand in for 63 (7 active highlighted)
    for i in range(14):
        r, c = divmod(i, 7)
        x = 600 + c * 62
        y = 360 + r * 48
        active = i < 7
        fill = "#fef3c7" if active else "#f1f5f9"
        stroke = "#f59e0b" if active else "#cbd5e1"
        L(f'<rect x="{x}" y="{y}" width="56" height="40" rx="6" fill="{fill}" stroke="{stroke}"/>')
        lab = f"R{i}" if i < 13 else "…"
        L(f'<text x="{x+28}" y="{y+17}" text-anchor="middle" class="node-text">{t(lab)}</text>')
        L(f'<text x="{x+28}" y="{y+32}" text-anchor="middle" class="node-text">{t("0.25x")}</text>')

    L(
        f'<text x="810" y="468" text-anchor="middle" class="node-text">{t("Activated = 1 shared + 7 routed = 8 x 0.25 = 2x FFN.")}</text>'
    )
    L(
        f'<text x="810" y="508" text-anchor="middle" class="node-title">{t("C(63,7) ~ 5.5e8 combinations")}</text>'
    )
    L(
        f'<text x="810" y="530" text-anchor="middle" class="note">{t("Paper: beats GShard 2B; matches GShard x1.5; Pile CE 1.808.")}</text>'
    )

    L(
        f'<text x="550" y="590" text-anchor="middle" class="note">{t("E1 trains both arms to 10B tokens on the frozen 40B mix. Hit = lower val BPB for arm B, no routing collapse.")}</text>'
    )
    L(
        f'<text x="550" y="610" text-anchor="middle" class="note">{t("Absolute paper numbers are out of scope as a target (GPT-2 50k vs 8k BPE; open mix vs internal corpus).")}</text>'
    )
    L("</svg>")
    write("dsmoe_vs_gshard.svg")


if __name__ == "__main__":
    fig()
