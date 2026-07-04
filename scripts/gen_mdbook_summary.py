#!/usr/bin/env python3
"""Generate mdBook SUMMARY.md from build_book READING_ORDER."""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BOOK = REPO / "《ds-技术报告》"
sys.path.insert(0, str(BOOK))

import build_book  # noqa: E402

SECTION_TITLES: dict[str, str] = {
    "00-前言": "前言",
    "01-总览": "01 总览",
    "02-基座架构": "02 基座架构",
    "03-后训练与R1": "03 后训练与 R1",
    "04-版本代际": "04 版本代际",
    "05-DSA稀疏注意力": "05 DSA 稀疏注意力",
    "06-推理基础设施": "06 推理基础设施",
    "07-Engram": "07 Engram",
    "08-外部解读": "08 外部解读",
    "09-附录": "09 附录",
}

TITLE_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)


def chapter_title(rel: str) -> str:
    path = BOOK / rel
    if path.is_file():
        m = TITLE_RE.search(path.read_text(encoding="utf-8"))
        if m:
            return m.group(1).strip()
    return Path(rel).stem


def main() -> None:
    lines = ["# Summary", ""]
    lines.append("[中文导读](00-前言/02-中文导读.md)")
    lines.append("")

    current_section: str | None = None
    order = ["00-前言/01-知识库导读.md", *build_book.READING_ORDER]
    seen: set[str] = set()

    for rel in order:
        if rel in seen or rel == "00-前言/02-中文导读.md":
            continue
        seen.add(rel)
        section = rel.split("/", 1)[0]
        if section != current_section:
            current_section = section
            lines.append("")
            lines.append(f"# {SECTION_TITLES.get(section, section)}")
        title = chapter_title(rel)
        lines.append(f"- [{title}]({rel})")

    out = BOOK / "SUMMARY.md"
    out.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"WRITE {out.relative_to(REPO)} ({len(seen) + 1} entries)")


if __name__ == "__main__":
    main()
