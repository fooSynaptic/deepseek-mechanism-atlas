#!/usr/bin/env python3
"""Replace <img src="*.svg"> blocks with inline <svg> for Markdown preview."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
from _paths import ENGRAM_DIAGRAMS  # noqa: E402

import re
import sys
from pathlib import Path

INLINE_BLOCK = re.compile(
    r'<div markdown="0" align="center">\s*\n+'
    r'<!-- inline: (?P<name>[^ ]+\.svg) -->\s*\n'
    r'<svg[\s\S]*?</svg>\s*\n+'
    r'</div>\s*\n+'
    r'\[图示详情\]\((?P<link>[^)]+)\)',
    re.MULTILINE,
)

IMG_BLOCK = re.compile(
    r'<img src="(?P<src>[^"]+\.svg)" alt="[^"]*"\s+width="\d+"/>\s*\n+'
    r'\[图示详情\]\((?P<link>[^)]+)\)',
    re.MULTILINE,
)


def inline_svg(svg_path: Path, link: str) -> str:
    raw = svg_path.read_text(encoding="utf-8")
    if raw.lstrip().startswith("<?xml"):
        raw = raw.split("?>", 1)[1].strip()
    # Asterisks in SVG text break Markdown preview when inlined (e.g. w0*v~_t).
    raw = raw.replace("*", "&#42;")
    return (
        f'<div markdown="0" align="center">\n\n'
        f"<!-- inline: {svg_path.name} -->\n"
        f"{raw}\n\n"
        f"</div>\n\n"
        f"[图示详情]({link})\n"
    )


def embed_in_markdown(md_path: Path) -> bool:
    text = md_path.read_text(encoding="utf-8")
    md_dir = md_path.parent
    changed = False

    def repl_inline(m: re.Match[str]) -> str:
        nonlocal changed
        changed = True
        link = m.group("link")
        svg_path = (md_dir / link).resolve()
        if not svg_path.is_file():
            raise SystemExit(f"missing SVG: {svg_path}")
        return inline_svg(svg_path, link)

    def repl_img(m: re.Match[str]) -> str:
        nonlocal changed
        changed = True
        src = m.group("src")
        link = m.group("link")
        svg_path = (md_dir / src).resolve()
        if not svg_path.is_file():
            raise SystemExit(f"missing SVG: {svg_path}")
        return inline_svg(svg_path, link)

    new_text = INLINE_BLOCK.sub(repl_inline, text)
    new_text = IMG_BLOCK.sub(repl_img, new_text)
    if changed:
        md_path.write_text(new_text, encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print("usage: embed_svg_md.py <file.md> ...", file=sys.stderr)
        return 1
    for p in argv:
        path = Path(p)
        if embed_in_markdown(path):
            print(f"embedded SVGs in {path}")
        else:
            print(f"no <img> blocks in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
