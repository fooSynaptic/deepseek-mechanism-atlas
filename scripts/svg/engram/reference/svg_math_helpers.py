"""Reusable SVG math text helpers (Engram diagram reference).

Canonical example: gen_engram_01d_svg.py -> engram-01d-multi-head-hash.svg
See reference/README.md and docs/material/meta/svg-diagram-math.md
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
from _paths import ENGRAM_DIAGRAMS  # noqa: E402

import html

# Math font stack for variables / subscripts (LaTeX-like in SVG preview)
MATH_FONT = "Cambria Math, STIX Two Math, Latin Modern Math, serif"
LABEL_FONT = "system-ui, -apple-system, sans-serif"

# Greek letters as numeric entities (safe in SVG + Markdown-adjacent pipelines)
PHI = "&#x3C6;"  # φ
ALPHA = "&#x3B1;"  # α
DELTA = "&#x394;"  # Δ (use sparingly; prefer spelled-out in labels)

SUB_SCALE = "0.72em"


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def sub(
    x: int,
    y: int,
    base: str,
    subscript: str,
    *,
    cls: str = "m",
    anchor: str | None = None,
    italic_base: bool = True,
) -> str:
    """Render base_sub as nested <tspan> (e.g. z + t,n,k -> z_{t,n,k})."""
    a = f' text-anchor="{anchor}"' if anchor else ""
    ib = ' font-style="italic"' if italic_base else ""
    return (
        f'  <text x="{x}" y="{y}" class="{cls}"{a}>'
        f"<tspan{ib}>{esc(base)}</tspan>"
        f'<tspan baseline-shift="sub" font-size="{SUB_SCALE}">{esc(subscript)}</tspan>'
        f"</text>\n"
    )


def plain(
    x: int,
    y: int,
    text: str,
    *,
    cls: str = "b",
    anchor: str | None = None,
    font_family: str | None = None,
    **attrs: str,
) -> str:
    a = f' text-anchor="{anchor}"' if anchor else ""
    ff = f' font-family="{font_family}"' if font_family else ""
    extra = "".join(f' {k}="{esc(v)}"' for k, v in attrs.items())
    return f'  <text x="{x}" y="{y}" class="{cls}"{a}{ff}{extra}>{esc(text)}</text>\n'


def math_line(
    x: int,
    y: int,
    parts: list[str],
    *,
    cls: str = "m",
    anchor: str = "middle",
) -> str:
    """Centered formula row built from pre-built tspan fragments."""
    body = "".join(parts)
    return f'  <text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}">{body}</text>\n'


def t_var(base: str, sub: str = "", *, italic: bool = True) -> str:
    ib = ' font-style="italic"' if italic else ""
    if sub:
        return (
            f"<tspan{ib}>{esc(base)}</tspan>"
            f'<tspan baseline-shift="sub" font-size="{SUB_SCALE}">{esc(sub)}</tspan>'
        )
    return f"<tspan{ib}>{esc(base)}</tspan>"


def t_greek(name: str, sub: str = "") -> str:
    entities = {"phi": PHI, "alpha": ALPHA}
    ch = entities.get(name, esc(name))
    if sub:
        return (
            f"<tspan font-style=\"italic\">{ch}</tspan>"
            f'<tspan baseline-shift="sub" font-size="{SUB_SCALE}">{esc(sub)}</tspan>'
        )
    return f'<tspan font-style="italic">{ch}</tspan>'


def default_math_styles() -> str:
    return f"""
      .t{{font-size:13px;font-weight:700;fill:#0f172a;text-anchor:middle;font-family:{LABEL_FONT}}}
      .s{{font-size:9px;fill:#64748b;text-anchor:middle;font-family:{LABEL_FONT}}}
      .b{{font-size:10px;fill:#334155;text-anchor:middle}}
      .m{{font-size:11px;fill:#334155;text-anchor:middle}}
      .fm{{font-size:9px;fill:#475569;text-anchor:middle}}
      .box{{rx:5;stroke-width:1.2}}
"""
