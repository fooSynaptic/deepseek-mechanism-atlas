#!/usr/bin/env python3
"""Restore <img> placeholders from inline SVG blocks (re-run embed after SVG edits)."""
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
    r'\[直接打开 SVG\]\((?P<link>[^)]+)\)',
    re.MULTILINE,
)

# legacy without markdown="0"
INLINE_BLOCK_LEGACY = re.compile(
    r'<div align="center">\s*\n+'
    r'<!-- inline: (?P<name>[^ ]+\.svg) -->\s*\n'
    r'<svg[\s\S]*?</svg>\s*\n+'
    r'</div>\s*\n+'
    r'\[直接打开 SVG\]\((?P<link>[^)]+)\)',
    re.MULTILINE,
)


def restore_img(md_path: Path) -> bool:
    text = md_path.read_text(encoding="utf-8")
    changed = False

    def repl(m: re.Match[str]) -> str:
        nonlocal changed
        changed = True
        link = m.group("link")
        name = Path(link).name
        alt = name.replace("-", " ").replace(".svg", "")
        w = "920" if "01f" in name or "02" in name else "860"
        return (
            f'<img src="{link}" alt="{alt}" width="{w}"/>\n\n'
            f"[直接打开 SVG]({link})\n"
        )

    for pat in (INLINE_BLOCK, INLINE_BLOCK_LEGACY):
        text, n = pat.subn(repl, text)
        if n:
            changed = True
    if changed:
        md_path.write_text(text, encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    for p in argv:
        path = Path(p)
        if restore_img(path):
            print(f"restored <img> in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
