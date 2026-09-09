#!/usr/bin/env python3
"""Render IsoFLOP design figures (English labels, XML-safe)."""

from __future__ import annotations

import math
from pathlib import Path

OUT = Path(__file__).resolve().parent


def e(s: str) -> str:
    return "".join(f"&#x{ord(c):X};" if ord(c) > 127 else c for c in s)


def esc(s: str) -> str:
    return (
        e(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


# After e(), ASCII is unchanged; escape XML specials on the raw string first.
def t(s: str) -> str:
    raw = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    return e(raw)


lines: list[str] = []


def L(s: str) -> None:
    lines.append(s)


def reset() -> None:
    lines.clear()


def write(name: str) -> None:
    dest = OUT / name
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", dest)


# ---------------------------------------------------------------------------
# Shared chart helpers
# ---------------------------------------------------------------------------

def defs_block() -> None:
    L("<defs>")
    L("  <style>")
    L("    .title { font-size: 20px; font-weight: 700; fill: #1a1a2e; }")
    L("    .subtitle { font-size: 12px; fill: #64748b; }")
    L("    .panel-title { font-size: 14px; font-weight: 700; fill: #1e293b; }")
    L("    .axis { font-size: 11px; fill: #64748b; }")
    L("    .axis-title { font-size: 11px; font-weight: 600; fill: #334155; }")
    L("    .tick { font-size: 10px; fill: #64748b; }")
    L("    .legend-text { font-size: 11px; fill: #334155; }")
    L("    .callout { font-size: 11px; fill: #9f1239; font-weight: 600; }")
    L("    .callout-ok { font-size: 11px; fill: #1d4ed8; font-weight: 600; }")
    L("    .note { font-size: 11px; fill: #475569; }")
    L("    .badge-text { font-size: 11px; font-weight: 600; fill: #fff; }")
    L("    .node-title { font-size: 13px; font-weight: 700; fill: #1e293b; }")
    L("    .node-text { font-size: 11.5px; fill: #475569; }")
    L("  </style>")
    L('  <marker id="arrow-blue" viewBox="0 0 10 8" refX="9" refY="4"')
    L('          markerWidth="8" markerHeight="6" orient="auto-start-reverse">')
    L('    <path d="M0,0 L10,4 L0,8 Z" fill="#3b82f6"/>')
    L("  </marker>")
    L('  <marker id="arrow-red" viewBox="0 0 10 8" refX="9" refY="4"')
    L('          markerWidth="8" markerHeight="6" orient="auto-start-reverse">')
    L('    <path d="M0,0 L10,4 L0,8 Z" fill="#e11d48"/>')
    L("  </marker>")
    L('  <marker id="arrow-gray" viewBox="0 0 10 8" refX="9" refY="4"')
    L('          markerWidth="8" markerHeight="6" orient="auto-start-reverse">')
    L('    <path d="M0,0 L10,4 L0,8 Z" fill="#64748b"/>')
    L("  </marker>")
    L("</defs>")


def panel(x, y, w, h, fill="#fff", stroke="#e2e8f0") -> None:
    L(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" fill="{fill}" stroke="{stroke}"/>')


def badge(x, y, w, h, fill, label) -> None:
    L(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}"/>')
    L(f'<text x="{x + w/2:.1f}" y="{y + h*0.68:.1f}" text-anchor="middle" class="badge-text">{t(label)}</text>')


def polyline(pts, color, width=2.2, dash=None, opacity=1.0) -> None:
    d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    L(
        f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="{width}"'
        f' stroke-linejoin="round" stroke-linecap="round" opacity="{opacity}"{extra}/>'
    )


def log_x(x, xmin, xmax, left, width) -> float:
    return left + (math.log10(x) - math.log10(xmin)) / (math.log10(xmax) - math.log10(xmin)) * width


def lin_y(y, ymin, ymax, top, height) -> float:
    return top + (ymax - y) / (ymax - ymin) * height


def axes(left, top, width, height, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel, xtick_sci=False) -> None:
    L(f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="#f8fafc" stroke="#e2e8f0"/>')
    for xv in xticks:
        x = log_x(xv, xmin, xmax, left, width)
        L(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+height}" stroke="#e2e8f0"/>')
        if xtick_sci:
            lab = f"{xv:.0e}".replace("+0", "").replace("+", "")
        elif xv >= 100:
            lab = f"10^{int(round(math.log10(xv)))}"
        else:
            lab = str(xv)
        L(f'<text x="{x:.1f}" y="{top+height+16}" text-anchor="middle" class="tick">{t(lab)}</text>')
    for yv in yticks:
        y = lin_y(yv, ymin, ymax, top, height)
        L(f'<line x1="{left}" y1="{y:.1f}" x2="{left+width}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        L(f'<text x="{left-8}" y="{y+3.5:.1f}" text-anchor="end" class="tick">{t(str(yv))}</text>')
    L(f'<rect x="{left}" y="{top}" width="{width}" height="{height}" fill="none" stroke="#cbd5e1"/>')
    L(
        f'<text x="{left+width/2:.1f}" y="{top+height+32}" text-anchor="middle" class="axis-title">{t(xlabel)}</text>'
    )
    L(
        f'<text x="{left-36}" y="{top+height/2:.1f}" text-anchor="middle" class="axis-title" '
        f'transform="rotate(-90 {left-36} {top+height/2:.1f})">{t(ylabel)}</text>'
    )


def paper_u(m, c, bpb0, m0):
    """Schematic U: valley at paper Formula 4 M_opt, depth falls with C."""
    m_opt = 0.1715 * (c ** 0.5243)
    z = (math.log10(m) - math.log10(m_opt)) / 0.45
    return bpb0 + 0.09 * z * z


def fig_paper_vs_pilot() -> None:
    reset()
    W, H = 1120, 980
    L('<?xml version="1.0" encoding="UTF-8"?>')
    L(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"')
    L('     font-family="PingFang SC,Microsoft YaHei,Helvetica Neue,Arial,sans-serif">')
    defs_block()
    L('<rect width="100%" height="100%" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>')
    title1 = "Paper Figure 4a claim vs this lab's TinyStories pilot"
    L(f'<text x="40" y="38" class="title">{t(title1)}</text>')
    L(
        f'<text x="40" y="58" class="subtitle">{t("Left is a schematic of DeepSeek-LLM (arXiv:2401.02954) Fig. 4a, not a scan of the paper. Right is measured val BPB.")}</text>'
    )

    # ----- left: paper schematic -----
    panel(24, 74, 528, 430)
    badge(40, 88, 168, 24, "#2563eb", "Paper claim  (schematic)")
    L(f'<text x="220" y="105" class="note">{t("fixed C = M·D, valley = M_opt")}</text>')
    left, top, cw, ch = 88, 130, 420, 300
    xmin, xmax, ymin, ymax = 8e7, 8e9, 0.70, 1.25
    axes(
        left, top, cw, ch, xmin, xmax, ymin, ymax,
        [1e8, 3e8, 1e9, 3e9],
        [0.75, 0.85, 0.95, 1.05, 1.15, 1.25],
        "non-embedding FLOPs / token  M  (log)",
        "val bits-per-byte",
    )
    paper_cs = [
        (1e17, 1.08, "#93c5fd", "C=1e17"),
        (3e17, 1.02, "#60a5fa", "C=3e17"),
        (1e18, 0.96, "#3b82f6", "C=1e18"),
        (3e18, 0.90, "#1d4ed8", "C=3e18"),
    ]
    ms = [10 ** x for x in [i / 40 * (math.log10(xmax) - math.log10(xmin)) + math.log10(xmin) for i in range(41)]]
    for c, b0, col, _lab in paper_cs:
        pts = []
        for m in ms:
            b = paper_u(m, c, b0, None)
            if ymin <= b <= ymax:
                pts.append((log_x(m, xmin, xmax, left, cw), lin_y(b, ymin, ymax, top, ch)))
        polyline(pts, col, 2.4)
        m_opt = 0.1715 * (c ** 0.5243)
        x = log_x(m_opt, xmin, xmax, left, cw)
        y = lin_y(paper_u(m_opt, c, b0, None), ymin, ymax, top, ch)
        L(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="{col}" stroke="#fff" stroke-width="1.2"/>')
    # valley arrow
    m_star = 0.1715 * (1e18 ** 0.5243)
    xs = log_x(m_star, xmin, xmax, left, cw)
    ys = lin_y(paper_u(m_star, 1e18, 0.96, None), ymin, ymax, top, ch)
    L(
        f'<line x1="{xs:.1f}" y1="{ys+18:.1f}" x2="{xs:.1f}" y2="{ys+48:.1f}" stroke="#1d4ed8" '
        f'stroke-width="1.6" marker-end="url(#arrow-blue)"/>'
    )
    L(f'<text x="{xs+8:.1f}" y="{ys+62:.1f}" class="callout-ok">{t("M_opt  (moves right as C grows)")}</text>')
    ly = 148
    for c, b0, col, lab in paper_cs:
        L(f'<rect x="400" y="{ly}" width="18" height="3" rx="1" fill="{col}"/>')
        L(f'<text x="424" y="{ly+5}" class="legend-text">{t(lab)}</text>')
        ly += 16
    L(f'<text x="288" y="478" text-anchor="middle" class="note">{t("Each IsoFLOP slice is U-shaped. Interior min = allocate more model as C grows.")}</text>')

    # ----- right: pilot -----
    panel(568, 74, 528, 430)
    badge(584, 88, 150, 24, "#e11d48", "Pilot  (measured)")
    L(f'<text x="746" y="105" class="note">{t("TinyStories, L=2048, unique D=462M")}</text>')
    left, top, cw, ch = 632, 130, 420, 300
    xmin, xmax, ymin, ymax = 8e7, 5e9, 0.35, 1.40
    axes(
        left, top, cw, ch, xmin, xmax, ymin, ymax,
        [1e8, 3e8, 1e9, 3e9],
        [0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
        "non-embedding FLOPs / token  M  (log)",
        "val bits-per-byte",
    )
    M = [1.203e8, 2.517e8, 4.522e8, 7.361e8, 1.118e9, 1.611e9, 2.013e9, 3.586e9]
    labs = list("abcdefgh")
    bpb_1e17 = [0.4565, 0.4766, 0.5460, 0.6973, 0.8344, 1.0251, 1.1832, 1.3028]
    bpb_3e17 = [0.4327, 0.4149, 0.4200, 0.4722, 0.6768, 0.7819, 0.7719, 1.1512]
    pts1 = [(log_x(m, xmin, xmax, left, cw), lin_y(b, ymin, ymax, top, ch)) for m, b in zip(M, bpb_1e17)]
    pts3 = [(log_x(m, xmin, xmax, left, cw), lin_y(b, ymin, ymax, top, ch)) for m, b in zip(M, bpb_3e17)]
    polyline(pts1, "#fb7185", 2.4)
    polyline(pts3, "#e11d48", 2.4)
    for (x, y), lab in zip(pts1, labs):
        L(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="#fb7185" stroke="#fff" stroke-width="1"/>')
    for (x, y), lab in zip(pts3, labs):
        L(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.6" fill="#e11d48" stroke="#fff" stroke-width="1"/>')
        if lab in ("a", "b", "h"):
            L(f'<text x="{x+6:.1f}" y="{y-6:.1f}" class="tick">{t(lab)}</text>')
    # monotone arrow
    x0, y0 = pts1[0]
    x1, y1 = pts1[-1]
    L(
        f'<path d="M{x0+12:.1f},{y0-8:.1f} Q{(x0+x1)/2:.1f},{y0-40:.1f} {x1-8:.1f},{y1-10:.1f}" '
        f'fill="none" stroke="#e11d48" stroke-width="1.6" marker-end="url(#arrow-red)"/>'
    )
    L(f'<text x="{x0+40:.1f}" y="{y0-44:.1f}" class="callout">{t("C=1e17: monotone  —  no valley")}</text>')
    xb, yb = pts3[1]
    L(f'<circle cx="{xb:.1f}" cy="{yb:.1f}" r="7" fill="none" stroke="#be123c" stroke-width="1.6"/>')
    L(f'<text x="{xb+10:.1f}" y="{yb+18:.1f}" class="callout">{t("C=3e17: weak U at b, then blows up")}</text>')
    ly = 148
    L(f'<rect x="944" y="{ly}" width="18" height="3" rx="1" fill="#fb7185"/>')
    L(f'<text x="968" y="{ly+5}" class="legend-text">{t("C=1e17  measured")}</text>')
    L(f'<rect x="944" y="{ly+16}" width="18" height="3" rx="1" fill="#e11d48"/>')
    L(f'<text x="968" y="{ly+21}" class="legend-text">{t("C=3e17  measured")}</text>')
    L(f'<text x="832" y="478" text-anchor="middle" class="note">{t("Left arm of the U is missing: even the smallest model already fits TinyStories.")}</text>')

    # ----- bottom: why -----
    panel(24, 520, 1072, 436, fill="#fff")
    badge(40, 536, 210, 24, "#0f172a", "Why the U collapsed")
    L(
        f'<text x="264" y="553" class="note">{t("IsoFLOP still spent C = M·D FLOPs. D was not unique tokens, and the distribution was too easy.")}</text>'
    )

    # three cause cards
    cards = [
        (40, 576, "#fef2f2", "#fecaca", "#be123c", "1  Toy domain",
         "TinyStories is a closed children-story",
         "distribution. ~8M params already reach",
         "BPB 0.46. There is no under-capacity",
         "regime, so the left arm of the U never",
         "appears."),
        (392, 576, "#fff7ed", "#fed7aa", "#c2410c", "2  Unique-token loop",
         "Only 462M unique train tokens. Jobs",
         "with D > 462M (small M, large C)",
         "replay the same stories. C=3e17 width a",
         "asked for 2.49B tokens ~ 5.4 epochs of",
         "the same 462M."),
        (744, 576, "#eff6ff", "#bfdbfe", "#1d4ed8", "3  Large-M starvation",
         "At fixed C, large M gets tiny D.",
         "Width h @ C=1e17 saw 28M tokens.",
         "The model is too big for that budget",
         "on this domain, so BPB climbs 0.46",
         "to 1.30. Only the right arm remains."),
    ]
    for x, y, bg, st, accent, title, *body in cards:
        L(f'<rect x="{x}" y="{y}" width="336" height="168" rx="8" fill="{bg}" stroke="{st}"/>')
        L(f'<rect x="{x}" y="{y}" width="336" height="28" rx="8" fill="{accent}"/>')
        L(f'<rect x="{x}" y="{y+20}" width="336" height="10" fill="{accent}"/>')
        L(f'<text x="{x+168}" y="{y+20}" text-anchor="middle" class="badge-text">{t(title)}</text>')
        yy = y + 52
        for line in body:
            L(f'<text x="{x+16}" y="{yy}" class="node-text">{t(line)}</text>')
            yy += 18

    # bar chart D vs unique
    L(f'<text x="40" y="772" class="node-title">{t("Scheduled D vs unique tokens  (C=3e17, the worse budget)")}</text>')
    bx, by, bw, bh = 40, 786, 1040, 140
    L(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="8" fill="#f8fafc" stroke="#e2e8f0"/>')
    D3 = [2493, 1192, 663, 408, 268, 186, 149, 84]  # millions
    unique = 462
    maxd = 2600
    gap = 8
    bar_w = 70
    origin = bx + 50
    for i, (lab, d) in enumerate(zip(labs, D3)):
        x = origin + i * (bar_w + 48)
        h = d / maxd * 96
        color = "#e11d48" if d > unique else "#64748b"
        L(f'<rect x="{x}" y="{by+118-h:.1f}" width="{bar_w}" height="{h:.1f}" rx="3" fill="{color}" opacity="0.85"/>')
        L(f'<text x="{x+bar_w/2:.1f}" y="{by+132}" text-anchor="middle" class="tick">{t(lab)}</text>')
        L(f'<text x="{x+bar_w/2:.1f}" y="{by+114-h:.1f}" text-anchor="middle" class="tick">{t(str(d)+"M")}</text>')
    y_u = by + 118 - unique / maxd * 96
    L(
        f'<line x1="{bx+20}" y1="{y_u:.1f}" x2="{bx+bw-20}" y2="{y_u:.1f}" stroke="#0f172a" '
        f'stroke-width="1.6" stroke-dasharray="6,3"/>'
    )
    L(f'<text x="{bx+bw-24}" y="{y_u-6:.1f}" text-anchor="end" class="axis-title">{t("unique = 462M  (loop above this line)")}</text>')
    L("</svg>")
    write("isoflop_paper_vs_pilot.svg")


def fig_v2_expected() -> None:
    reset()
    W, H = 1120, 900
    L('<?xml version="1.0" encoding="UTF-8"?>')
    L(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"')
    L('     font-family="PingFang SC,Microsoft YaHei,Helvetica Neue,Arial,sans-serif">')
    defs_block()
    L('<rect width="100%" height="100%" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>')
    L(f'<text x="40" y="38" class="title">{t("v2 IsoFLOP: expected shape under paper Formula 4")}</text>')
    L(
        f'<text x="40" y="58" class="subtitle">{t("Dashed U is Formula 4 on this C range, not a BPB forecast. The 1.1B-7B grid sits on the right arm unless this mix is more model-hungry than the paper.")}</text>'
    )

    panel(24, 74, 1072, 430)
    badge(40, 88, 132, 24, "#0f766e", "Wave A  expected")
    L(f'<text x="184" y="105" class="note">{t("s1-s6 at L=4096  ·  C in {3e18, 1e19, 3e19}  ·  4.0B unique open-mix tokens")}</text>')

    left, top, cw, ch = 88, 128, 820, 310
    xmin, xmax, ymin, ymax = 4e8, 6e10, 0.70, 1.22
    axes(
        left, top, cw, ch, xmin, xmax, ymin, ymax,
        [1e9, 3e9, 1e10, 3e10],
        [0.75, 0.85, 0.95, 1.05, 1.15],
        "non-embedding FLOPs / token  M  (log)",
        "val bits-per-byte  (schematic)",
    )
    Ms = {
        "s1": 8.858e9,
        "s2": 1.434e10,
        "s3": 2.159e10,
        "s4": 3.083e10,
        "s5": 4.228e10,
        "s6": 4.510e10,
    }
    # shade this lab's grid (right of paper valleys)
    x_s1 = log_x(Ms["s1"], xmin, xmax, left, cw)
    x_s6 = log_x(Ms["s6"], xmin, xmax, left, cw)
    L(
        f'<rect x="{x_s1:.1f}" y="{top}" width="{x_s6-x_s1:.1f}" height="{ch}" '
        f'fill="#0f766e" opacity="0.06"/>'
    )
    L(
        f'<text x="{(x_s1+x_s6)/2:.1f}" y="{top+18}" text-anchor="middle" class="tick">{t("this lab grid  (1.1B-7B)")}</text>'
    )

    cs = [
        (3e18, "#5eead4", "C=3e18"),
        (1e19, "#14b8a6", "C=1e19"),
        (3e19, "#0f766e", "C=3e19"),
    ]
    m_plot = [
        10 ** x
        for x in [
            i / 60 * (math.log10(xmax) - math.log10(xmin)) + math.log10(xmin) for i in range(61)
        ]
    ]

    def exp_bpb(m, c):
        m_opt = 0.1715 * (c ** 0.5243)
        floor = {3e18: 0.97, 1e19: 0.89, 3e19: 0.81}[c]
        z = (math.log10(m) - math.log10(m_opt)) / 0.38
        return floor + 0.065 * z * z

    for c, col, lab in cs:
        pts = []
        for m in m_plot:
            b = exp_bpb(m, c)
            if b > ymax:
                continue
            pts.append((log_x(m, xmin, xmax, left, cw), lin_y(min(b, ymax), ymin, ymax, top, ch)))
        polyline(pts, col, 2.5, dash="7,4")
        m_opt = 0.1715 * (c ** 0.5243)
        if xmin <= m_opt <= xmax:
            x = log_x(m_opt, xmin, xmax, left, cw)
            y = lin_y(exp_bpb(m_opt, c), ymin, ymax, top, ch)
            L(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top+ch}" stroke="{col}" '
                f'stroke-width="1" stroke-dasharray="3,3" opacity="0.55"/>'
            )
            L(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{col}" stroke="#fff" stroke-width="1.4"/>')
        if c == 3e19:
            x = log_x(m_opt, xmin, xmax, left, cw)
            L(f'<text x="{x:.1f}" y="{top-6}" text-anchor="middle" class="tick">{t("paper M_opt")}</text>')

    for name, m in Ms.items():
        x = log_x(m, xmin, xmax, left, cw)
        L(f'<line x1="{x:.1f}" y1="{top+ch}" x2="{x:.1f}" y2="{top+ch+6}" stroke="#334155"/>')
        L(f'<text x="{x:.1f}" y="{top+ch+18}" text-anchor="middle" class="tick">{t(name)}</text>')
        for c, col, _ in cs:
            y = lin_y(min(exp_bpb(m, c), ymax), ymin, ymax, top, ch)
            L(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{col}" stroke="#fff" stroke-width="0.8"/>')

    ly = 148
    for c, col, lab in cs:
        L(f'<rect x="930" y="{ly}" width="22" height="3" rx="1" fill="{col}"/>')
        L(f'<text x="958" y="{ly+5}" class="legend-text">{t(lab)}</text>')
        ly += 16
    L(f'<text x="930" y="{ly+10}" class="note">{t("s5 = paper 7B width")}</text>')
    L(f'<text x="930" y="{ly+26}" class="note">{t("30L / d=4096 / 32H")}</text>')
    L(f'<text x="930" y="{ly+48}" class="callout-ok">{t("A  Formula 4 holds:")}</text>')
    L(f'<text x="930" y="{ly+64}" class="note">{t("s1 best on every C")}</text>')
    L(f'<text x="930" y="{ly+80}" class="note">{t("(right arm only)")}</text>')
    L(f'<text x="930" y="{ly+102}" class="callout">{t("B  Mix wants more model:")}</text>')
    L(f'<text x="930" y="{ly+118}" class="note">{t("interior U on s1-s6")}</text>')

    L(
        f'<text x="560" y="478" text-anchor="middle" class="note">{t("Paper M_opt at these C is 0.84B / 1.57B / 2.79B FLOPs/tok, all left of s1 (8.86B). A visible U on 1.1B-7B would mean a larger than 0.52.")}</text>'
    )

    # bottom: unique floor + protocol
    panel(24, 520, 528, 352)
    badge(40, 536, 190, 24, "#1d4ed8", "Unique-token floor  (met)")
    L(f'<text x="40" y="580" class="node-title">{t("Scheduled D vs 4.0B unique train tokens")}</text>')
    # s1 is the max D at each C
    rows = [
        ("C=3e18", "s1 339M", "s5 71M", False),
        ("C=1e19", "s1 1.13B", "s5 237M", False),
        ("C=3e19", "s1 3.39B", "s5 710M", False),
    ]
    yy = 600
    L(f'<rect x="40" y="592" width="496" height="1" fill="#e2e8f0"/>')
    for lab, a, b, _ in rows:
        L(f'<text x="56" y="{yy+18}" class="node-text-bold" font-weight="600" fill="#1e293b">{t(lab)}</text>')
        L(f'<text x="160" y="{yy+18}" class="node-text">{t("max D  " + a + "   ·   7B D  " + b)}</text>')
        L(f'<text x="480" y="{yy+18}" text-anchor="end" fill="#15803d" font-size="12" font-weight="700">{t("D < 4.0B")}</text>')
        yy += 28
        L(f'<rect x="40" y="{yy}" width="496" height="1" fill="#f1f5f9"/>')
    L(f'<text x="40" y="710" class="node-text">{t("Pilot failed this test (D up to 2.49B on 462M unique).")}</text>')
    L(f'<text x="40" y="730" class="node-text">{t("Mix: FineWeb-Edu + Wiki EN/ZH + math + books + CC WET.")}</text>')
    L(f'<text x="40" y="750" class="node-text">{t("Code / C4 slices missed; Wave A can still run.")}</text>')
    L(f'<text x="40" y="784" class="node-text">{t("Pilot-style fail: BPB ~0.4 on the smallest width")}</text>')
    L(f'<text x="40" y="802" class="node-text">{t("(memorizing an easy domain) plus looping D.")}</text>')
    L(f'<text x="40" y="834" class="node-text">{t("Do not fit Formula 4 unless a valley is interior or")}</text>')
    L(f'<text x="40" y="852" class="node-text">{t("s1-best plus BPB falling with C at fixed width.")}</text>')

    panel(568, 520, 528, 352)
    badge(584, 536, 150, 24, "#7c3aed", "How v2 is run")
    steps = [
        ("Wave A", "18 IsoFLOP jobs, packed 8+8+4 GPUs.", "s5/s6 = 8-GPU FSDP; s1 = 1 GPU."),
        ("Read C2", "Plot val BPB vs M at each C.", "Interior U, or s1-best + BPB falls with C."),
        ("Then C3", "Fit a only if a valley is identified.", "Paper a=0.52; band [0.35, 0.70]."),
        ("Wave B", "Formula 1 cross at s3, C=3e18.", "Center inside 0.25% of the min BPB."),
    ]
    y = 576
    for i, (title, l1, l2) in enumerate(steps):
        L(f'<rect x="584" y="{y}" width="496" height="68" rx="8" fill="#fff" stroke="#ddd6fe"/>')
        L(f'<rect x="584" y="{y}" width="496" height="24" rx="8" fill="#ede9fe"/>')
        L(f'<rect x="584" y="{y+16}" width="496" height="10" fill="#ede9fe"/>')
        L(f'<text x="832" y="{y+17}" text-anchor="middle" class="node-title" fill="#5b21b6">{t(f"{i+1}  {title}")}</text>')
        L(f'<text x="600" y="{y+42}" class="node-text">{t(l1)}</text>')
        L(f'<text x="600" y="{y+58}" class="node-text">{t(l2)}</text>')
        y += 76

    L("</svg>")
    write("isoflop_v2_expected.svg")


def fig_measured_l4() -> None:
    """Measured IsoFLOP after L4: C=3e18 u0-u3 and C=1e19 u1-u3 (40B mix)."""
    reset()
    W, H = 1120, 720
    L('<?xml version="1.0" encoding="UTF-8"?>')
    L(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"')
    L('     font-family="Helvetica Neue,Arial,sans-serif">')
    defs_block()
    L('<rect width="100%" height="100%" rx="12" fill="#f8fafc" stroke="#e2e8f0"/>')
    L(f'<text x="40" y="38" class="title">{t("Measured IsoFLOP after L4 (left of t0)")}</text>')
    L(
        f'<text x="40" y="58" class="subtitle">{t("Val BPB vs M. Solid L4 ticks share the frozen 40B mix. Dashed t0-s6 joins are older snapshots. u3@3e19 was stopped; not plotted.")}</text>'
    )

    c318 = [
        ("u0", 1.769e8, 1.0610, "l4"),
        ("u1", 3.523e8, 1.0010, "l4"),
        ("u2", 6.095e8, 0.9850, "l4"),
        ("u3", 8.022e8, 0.9798, "l4"),
        ("t0", 9.626e8, 1.0359, "prior"),
        ("t1", 1.629e9, 1.0574, "prior"),
        ("t2", 2.768e9, 1.0995, "prior"),
        ("t3", 3.831e9, 2.0949, "prior"),
        ("t4", 5.889e9, 2.2085, "prior"),
        ("s1", 8.858e9, 2.2323, "prior"),
        ("s2", 1.434e10, 2.3058, "prior"),
        ("s3", 2.159e10, 2.1990, "prior"),
        ("s4", 3.083e10, 2.2648, "prior"),
        ("s5", 4.228e10, 2.3727, "prior"),
        ("s6", 4.510e10, 2.4204, "prior"),
    ]
    c1e19 = [
        ("u1", 3.523e8, 0.9621, "l4"),
        ("u2", 6.095e8, 0.9154, "l4"),
        ("u3", 8.022e8, 0.8956, "l4"),
        ("t0", 9.626e8, 0.7874, "prior"),
        ("t1", 1.629e9, 0.7774, "prior"),
        ("t2", 2.768e9, 0.9833, "prior"),
        ("t3", 3.831e9, 0.9817, "prior"),
        ("t4", 5.889e9, 1.0664, "prior"),
        ("s1", 8.858e9, 2.1215, "prior"),
        ("s2", 1.434e10, 2.2284, "prior"),
        ("s3", 2.159e10, 2.2163, "prior"),
        ("s4", 3.083e10, 2.2736, "prior"),
        ("s5", 4.228e10, 2.2457, "prior"),
        ("s6", 4.510e10, 2.1996, "prior"),
    ]
    c3e19 = [
        ("t0", 9.626e8, 0.8505),
        ("t1", 1.629e9, 0.8227),
        ("t2", 2.768e9, 0.7146),
        ("t3", 3.831e9, 0.7326),
        ("t4", 5.889e9, 0.7607),
        ("s1", 8.858e9, 0.9504),
        ("s2", 1.434e10, 2.0341),
        ("s3", 2.159e10, 2.1713),
        ("s4", 3.083e10, 2.2393),
        ("s5", 4.228e10, 2.2279),
        ("s6", 4.510e10, 2.2357),
    ]
    m_opt = 0.1715 * (3e18 ** 0.5243)

    def xy(m, b, left, top, cw, ch, xmin, xmax, ymin, ymax):
        return (
            log_x(m, xmin, xmax, left, cw),
            lin_y(b, ymin, ymax, top, ch),
        )

    # ----- left: family -----
    panel(24, 74, 528, 520)
    badge(40, 88, 200, 24, "#1d4ed8", "Family  (three C)")
    left, top, cw, ch = 88, 130, 420, 390
    xmin, xmax, ymin, ymax = 1.4e8, 5.2e10, 0.65, 2.50
    axes(
        left, top, cw, ch, xmin, xmax, ymin, ymax,
        [3e8, 1e9, 3e9, 1e10, 3e10],
        [0.7, 1.0, 1.3, 1.6, 1.9, 2.2, 2.5],
        "non-embedding FLOPs / token  M  (log)",
        "val bits-per-byte",
    )
    xopt = log_x(m_opt, xmin, xmax, left, cw)
    L(
        f'<line x1="{xopt:.1f}" y1="{top}" x2="{xopt:.1f}" y2="{top+ch}" '
        f'stroke="#94a3b8" stroke-width="1.2" stroke-dasharray="4 3"/>'
    )
    L(
        f'<text x="{xopt+6:.1f}" y="{top+14}" class="tick">{t("paper M_opt@3e18")}</text>'
    )

    def draw_split(rows, solid, dashed, r_l4=4.2, r_pr=3.2):
        l4 = [r for r in rows if r[3] == "l4"]
        prior = [r for r in rows if r[3] == "prior"]
        pts_l4 = [xy(m, b, left, top, cw, ch, xmin, xmax, ymin, ymax) for _, m, b, _ in l4]
        pts_pr = [xy(m, b, left, top, cw, ch, xmin, xmax, ymin, ymax) for _, m, b, _ in prior]
        if pts_l4:
            polyline(pts_l4, solid, 2.6)
        if pts_pr:
            polyline(pts_pr, dashed, 2.2, dash="6 4")
        if pts_l4 and pts_pr:
            polyline([pts_l4[-1], pts_pr[0]], "#94a3b8", 1.6, dash="3 3")
        for x, y in pts_l4:
            L(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r_l4}" fill="{solid}" stroke="#fff" stroke-width="1.2"/>')
        for x, y in pts_pr:
            L(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r_pr}" fill="{dashed}" stroke="#fff" stroke-width="1"/>')

    draw_split(c318, "#1d4ed8", "#93c5fd")
    draw_split(c1e19, "#0369a1", "#7dd3fc", r_l4=4.0, r_pr=3.0)
    pts3 = [xy(m, b, left, top, cw, ch, xmin, xmax, ymin, ymax) for _, m, b in c3e19]
    polyline(pts3, "#15803d", 2.2)
    for x, y in pts3:
        L(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="#15803d" stroke="#fff" stroke-width="1"/>')

    ly = 148
    for col, lab in (
        ("#1d4ed8", "C=3e18 L4 u0-u3  (40B mix)"),
        ("#93c5fd", "C=3e18 t0-s6  (4B mix, dashed)"),
        ("#0369a1", "C=1e19 L4 u1-u3  (40B mix)"),
        ("#7dd3fc", "C=1e19 t0-s6  (10.6B mix, dashed)"),
        ("#15803d", "C=3e19 t0-s6  (mixed snapshots)"),
    ):
        L(f'<rect x="96" y="{ly}" width="18" height="3" rx="1" fill="{col}"/>')
        L(f'<text x="120" y="{ly+5}" class="legend-text">{t(lab)}</text>')
        ly += 16
    L(
        f'<text x="288" y="568" text-anchor="middle" class="note">{t("Right cliff is unchanged. L4 fills the left of C=3e18 and C=1e19.")}</text>'
    )

    # ----- right: zoom C=3e18 + C=1e19 left -----
    panel(568, 74, 528, 520)
    badge(584, 88, 280, 24, "#0f172a", "Left-arm zoom  (3e18 + 1e19)")
    left, top, cw, ch = 632, 130, 420, 390
    xmin, xmax, ymin, ymax = 1.5e8, 3.2e9, 0.74, 1.14
    axes(
        left, top, cw, ch, xmin, xmax, ymin, ymax,
        [2e8, 4e8, 8e8, 1.6e9, 3e9],
        [0.78, 0.86, 0.94, 1.02, 1.10],
        "M  (log)",
        "val bits-per-byte",
        True,
    )
    m_opt_1e19 = 0.1715 * (1e19 ** 0.5243)
    xopt = log_x(m_opt, xmin, xmax, left, cw)
    xopt2 = log_x(m_opt_1e19, xmin, xmax, left, cw)
    L(
        f'<line x1="{xopt:.1f}" y1="{top}" x2="{xopt:.1f}" y2="{top+ch}" '
        f'stroke="#93c5fd" stroke-width="1.2" stroke-dasharray="4 3"/>'
    )
    L(
        f'<line x1="{xopt2:.1f}" y1="{top}" x2="{xopt2:.1f}" y2="{top+ch}" '
        f'stroke="#7dd3fc" stroke-width="1.2" stroke-dasharray="4 3"/>'
    )
    L(
        f'<text x="{xopt-6:.1f}" y="{top+14}" text-anchor="end" class="tick">{t("M_opt@3e18")}</text>'
    )
    L(
        f'<text x="{xopt2+6:.1f}" y="{top+14}" class="tick">{t("M_opt@1e19")}</text>'
    )

    def zoom_split(rows, solid, dashed, label_l4=True):
        z = [r for r in rows if r[0] in ("u0", "u1", "u2", "u3", "t0", "t1", "t2")]
        z_l4 = [r for r in z if r[3] == "l4"]
        z_pr = [r for r in z if r[3] == "prior"]
        p_l4 = [xy(m, b, left, top, cw, ch, xmin, xmax, ymin, ymax) for _, m, b, _ in z_l4]
        p_pr = [xy(m, b, left, top, cw, ch, xmin, xmax, ymin, ymax) for _, m, b, _ in z_pr]
        if p_l4:
            polyline(p_l4, solid, 2.6)
        if p_l4 and p_pr:
            polyline([p_l4[-1], p_pr[0]], "#94a3b8", 1.4, dash="3 3")
        if p_pr:
            polyline(p_pr, dashed, 2.0, dash="6 4")
        for (x, y), row in zip(p_l4, z_l4):
            lab, _m, bpb, _w = row
            L(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.6" fill="{solid}" stroke="#fff" stroke-width="1.3"/>')
            if label_l4:
                L(f'<text x="{x+7:.1f}" y="{y-8:.1f}" class="legend-text">{t(f"{lab} {bpb:.3f}")}</text>')
        for (x, y), row in zip(p_pr, z_pr):
            L(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.4" fill="{dashed}" stroke="#fff" stroke-width="1"/>')
        return p_l4, z_l4

    p318, z318 = zoom_split(c318, "#1d4ed8", "#64748b", label_l4=True)
    p119, z119 = zoom_split(c1e19, "#0369a1", "#38bdf8", label_l4=False)
    for (x, y), row in zip(p119, z119):
        lab, _m, bpb, _w = row
        L(f'<text x="{x+7:.1f}" y="{y+14:.1f}" class="tick">{t(f"{lab} {bpb:.3f}")}</text>')
    xv, yv = p318[-1]
    L(f'<circle cx="{xv:.1f}" cy="{yv:.1f}" r="8" fill="none" stroke="#1d4ed8" stroke-width="1.6"/>')
    L(f'<text x="{xv-10:.1f}" y="{yv+26:.1f}" text-anchor="end" class="callout-ok">{t("3e18 min u3  0.980")}</text>')
    xv2, yv2 = p119[-1]
    L(f'<circle cx="{xv2:.1f}" cy="{yv2:.1f}" r="8" fill="none" stroke="#0369a1" stroke-width="1.6"/>')
    L(f'<text x="{xv2-10:.1f}" y="{yv2-16:.1f}" text-anchor="end" class="callout-ok">{t("1e19 still falling at u3  0.896")}</text>')
    L(
        f'<text x="832" y="548" text-anchor="middle" class="note">{t("3e18 left lift u0-u3 = +0.081. 1e19 left lift u1-u3 = +0.066, still left of paper M_opt.")}</text>'
    )
    L(
        f'<text x="832" y="568" text-anchor="middle" class="note">{t("Dashed t0-t2 are older snapshots. u3@3e19 was stopped at 29% of D; not a grid point.")}</text>'
    )

    L("</svg>")
    write("isoflop_measured_l4.svg")


if __name__ == "__main__":
    for c in (3e18, 1e19, 3e19):
        print(f"paper M_opt(C={c:.0e}) = {0.1715 * (c ** 0.5243):.3e}")
    fig_paper_vs_pilot()
    fig_v2_expected()
    fig_measured_l4()
