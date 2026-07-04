#!/usr/bin/env python3
"""Strip non-essential parenthetical meta-comments from docs/*.md."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

# Meta-only parentheticals (full-width or half-width parens).
STRIP_RES = [
    re.compile(r"[（(]论文[^）)]*[）)]"),
    re.compile(r"[（(]原文[^）)]*[）)]"),
    re.compile(r"[（(]社区[^）)]*[）)]"),
    re.compile(r"[（(]已合并[）)]"),
    re.compile(r"[（(]非阅读入口[）)]"),
    re.compile(r"[（(]对应原文[^）)]*[）)]"),
    re.compile(r"[（(]Raschka 整理[）)]"),
    re.compile(r"/ Raschka 整理"),
    re.compile(r"[（(]DeepSeek 推理加速 · 唯一入口[）)]"),
    re.compile(r"[（(]arXiv:[^）)]*[）)]"),  # in titles when redundant
]

# Keep substantive parens: math, links, short disambiguation like （非xxx） with 非
KEEP_HINT = re.compile(
    r"[（(]"
    r"(?:非(?!论文)|约 |仅 |可选|Classical|State|inactive|PD |GPU|CPU|EN\)|"
    r"0\\s*\\le|k\{|\\$|§\d|见 |含 |相对 |对照 |Figure \d+ —)"
)


def clean_line(line: str) -> str:
    if KEEP_HINT.search(line):
        # Still strip paper meta even if line has keep hints, unless only keep part
        pass
    out = line
    for pat in STRIP_RES:
        out = pat.sub("", out)
    # Collapse double spaces left in headings / prose
    out = re.sub(r"  +", " ", out)
    out = re.sub(r" +([，。；：、])", r"\1", out)
    out = re.sub(r"： +", "：", out)
    out = re.sub(r" +#", " #", out)
    return out.rstrip() + ("\n" if line.endswith("\n") else "")


def main() -> None:
    n_files = n_lines = 0
    for path in sorted(DOCS.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        new_lines = [clean_line(ln) if ln.strip() else ln for ln in lines]
        new_text = "".join(new_lines)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            n_files += 1
            n_lines += sum(1 for a, b in zip(lines, new_lines) if a != b)
    print(f"OK strip_redundant_parens: {n_files} files, {n_lines} lines touched")


if __name__ == "__main__":
    main()
