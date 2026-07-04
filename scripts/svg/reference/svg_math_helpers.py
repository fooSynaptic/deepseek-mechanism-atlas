"""Reusable SVG math text helpers (Engram diagram reference).

Canonical example: gen_engram_01d_svg.py -> engram-01d-multi-head-hash.svg
See reference/README.md and docs/material/meta/svg-diagram-math.md
"""
from __future__ import annotations

import html

# Math font stack for variables / subscripts (LaTeX-like in SVG preview)
MATH_FONT = "Cambria Math, STIX Two Math, Latin Modern Math, serif"
LABEL_FONT = "system-ui, -apple-system, sans-serif"

# Greek letters as numeric entities (safe in SVG + Markdown-adjacent pipelines)
PHI = "&#x3C6;"  # φ
ALPHA = "&#x3B1;"  # α
DELTA = "&#x394;"  # Δ (use sparingly; prefer spelled-out in labels)

SUB_SCALE = "0.72em"
SUB_DY = "0.35em"


def _sub(sub: str) -> str:
    """Subscript tspan; explicit math font + dy reset for Markdown SVG preview."""
    return (
        f'<tspan dy="{SUB_DY}" font-size="{SUB_SCALE}" font-family="{MATH_FONT}">'
        f"{esc(sub)}</tspan>"
        f'<tspan dy="-{SUB_DY}" font-size="1em"></tspan>'
    )


def _sup(sup: str) -> str:
    return (
        f'<tspan dy="-{SUB_DY}" font-size="{SUB_SCALE}" font-family="{MATH_FONT}">'
        f"{esc(sup)}</tspan>"
        f'<tspan dy="{SUB_DY}" font-size="1em"></tspan>'
    )


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
        f"{_sub(subscript)}"
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


def blt_math(x: int, y: int, parts: list[str]) -> str:
    """Left-aligned mixed label + math row (diagram bullet lines)."""
    body = "".join(parts)
    return f'  <text x="{x}" y="{y}" class="bl fm" text-anchor="start">{body}</text>\n'


def t_var(base: str, sub: str = "", *, italic: bool = True) -> str:
    ib = ' font-style="italic"' if italic else ""
    ff = f' font-family="{MATH_FONT}"'
    if sub:
        return f"<tspan{ib}{ff}>{esc(base)}</tspan>{_sub(sub)}"
    return f"<tspan{ib}{ff}>{esc(base)}</tspan>"


def t_greek(name: str, sub: str = "") -> str:
    entities = {"phi": PHI, "alpha": ALPHA}
    ch = entities.get(name, esc(name))
    if sub:
        return f'<tspan font-style="italic" font-family="{MATH_FONT}">{ch}</tspan>{_sub(sub)}'
    return f'<tspan font-style="italic" font-family="{MATH_FONT}">{ch}</tspan>'


def t_supsub(base: str, sub: str = "", sup: str = "", *, italic: bool = True) -> str:
    """e.g. h_t^{(k)} -> base h, sub t, sup (k)."""
    ib = ' font-style="italic"' if italic else ""
    ff = f' font-family="{MATH_FONT}"'
    out = f"<tspan{ib}{ff}>{esc(base)}</tspan>"
    if sub:
        out += _sub(sub)
    if sup:
        out += _sup(sup)
    return out


def t_emb(xsub: str) -> str:
    return f"<tspan>Emb(</tspan>{t_var('x', xsub)}<tspan>)</tspan>"


def t_rmsnorm(inner: str) -> str:
    return f"<tspan>RMSNorm(</tspan>{inner}<tspan>)</tspan>"


def t_trm(k: str) -> str:
    return f"<tspan font-family=\"{MATH_FONT}\">TRM</tspan>{_sub(k)}"


def t_loss(sub: str = "") -> str:
    if sub:
        return f'<tspan font-family="{MATH_FONT}">L</tspan>{_sub(sub)}'
    return f'<tspan font-family="{MATH_FONT}">L</tspan>'


def t_lambda(sub: str = "k") -> str:
    return f'<tspan font-family="{MATH_FONT}">&#x3BB;</tspan>{_sub(sub)}'


def t_loss_mtp(k: str = "k") -> str:
    return (
        f'<tspan font-family="{MATH_FONT}">L</tspan>'
        f"{_sub('MTP')}"
        f"{_sup(f'({k})')}"
    )


def t_outhead() -> str:
    return "<tspan>OutHead</tspan>"


def t_softmax(inner: str) -> str:
    return f"<tspan>softmax(</tspan>{inner}<tspan>)</tspan>"


def t_dot() -> str:
    return f'<tspan font-family="{MATH_FONT}"> &#x22C5; </tspan>'


def t_times() -> str:
    return f'<tspan font-family="{MATH_FONT}">&#xD7;</tspan>'


def t_arrow() -> str:
    return f'<tspan font-family="{MATH_FONT}"> &#x2192; </tspan>'


def t_sum(sub: str = "") -> str:
    out = f'<tspan font-family="{MATH_FONT}">&#x2211;</tspan>'
    if sub:
        out += _sub(sub)
    return out


def t_relu(inner: str) -> str:
    return (
        f'<tspan font-family="{MATH_FONT}">ReLU(</tspan>'
        f"{inner}"
        f'<tspan font-family="{MATH_FONT}">)</tspan>'
    )


def t_cn(text: str) -> str:
    """Chinese / sans label fragment inside a mixed math row."""
    return f'<tspan font-family="{LABEL_FONT}" font-style="normal">{esc(text)}</tspan>'


def t_idx(sub: str) -> str:
    """Indexer score I_{t,s}."""
    return t_var("I", sub)


def t_srow(val: str) -> str:
    """History slot label: s = 1 (not s subscript 1)."""
    return t_cn(f"s = {val}  ")


def t_c(sup: str, sub: str = "") -> str:
    """c^{KV}, c_t^{KV}."""
    return t_supsub("c", sub, sup=sup)


def t_w(sup: str, sub: str = "") -> str:
    """W^{DQ}, W^{UK}."""
    return t_supsub("W", sub, sup=sup)


def t_in() -> str:
    return f'<tspan font-family="{MATH_FONT}"> &#x2208; </tspan>'


def t_rope(inner: str) -> str:
    return f"<tspan>RoPE(</tspan>{inner}<tspan>)</tspan>"


def t_concat_norms(h_part: str, emb_part: str) -> str:
    return (
        f"{t_rmsnorm(h_part)}"
        f"<tspan> ; </tspan>"
        f"{t_rmsnorm(emb_part)}"
    )


def default_math_styles() -> str:
    return f"""
      .t{{font-size:13px;font-weight:700;fill:#0f172a;text-anchor:middle;font-family:{LABEL_FONT}}}
      .s{{font-size:9px;fill:#64748b;text-anchor:middle;font-family:{LABEL_FONT}}}
      .b{{font-size:10px;fill:#334155;text-anchor:middle;font-family:{LABEL_FONT}}}
      .lb{{font-size:10px;font-weight:600;fill:#222;text-anchor:middle;font-family:{LABEL_FONT}}}
      .dt{{font-size:9px;fill:#555;text-anchor:middle;font-family:{LABEL_FONT}}}
      .an{{font-size:9px;fill:#2563eb;text-anchor:middle;font-family:{LABEL_FONT}}}
      .st{{font-size:10px;fill:#666;text-anchor:middle;font-family:{LABEL_FONT}}}
      .m{{font-size:12px;fill:#1e293b;text-anchor:middle;font-family:{MATH_FONT}}}
      .fm{{font-size:10px;fill:#334155;text-anchor:middle;font-family:{MATH_FONT}}}
      .box{{rx:5;stroke-width:1.2}}
      .m tspan{{font-family:{MATH_FONT}}}
      .fm tspan{{font-family:{MATH_FONT}}}
      text.fm.t{{font-size:13px;font-weight:700;fill:#0f172a}}
      text.fm.lb{{font-size:10px;font-weight:600;fill:#222}}
      text.fm.dt{{font-size:9px;fill:#555}}
      text.fm.st{{font-size:10px;fill:#666}}
      text.fm.an{{font-size:9px;fill:#2563eb}}
      .bl{{font-size:10px;fill:#555;text-anchor:start;font-family:{LABEL_FONT}}}
      .bl.fm{{font-family:{MATH_FONT};fill:#555;text-anchor:start}}
      .bl.fm tspan{{font-family:{MATH_FONT}}}
"""
