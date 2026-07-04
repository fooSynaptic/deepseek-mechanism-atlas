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
DISPLAY_MATH_RE = re.compile(r"\$\$[^$]+\$\$")
INLINE_MATH_RE = re.compile(r"\$([^$]+)\$")
EMPTY_PARENS_RE = re.compile(r"\s*[（(]\s*[）)]")


def _plain_math(expr: str) -> str:
    """目录用：把简单 LaTeX 收成可读 ASCII/符号。"""
    s = expr.strip()
    s = s.replace(r"\cdot", "·").replace(r"\times", "×")
    s = re.sub(r"\\mathrm\{([^}]+)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\s*", "", s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s*·\s*", "·", s)
    return re.sub(r"\s+", " ", s).strip()


def sanitize_toc_title(title: str, fallback: str = "") -> str:
    """mdBook 侧栏不跑 KaTeX；目录标题将 $...$ 收成 plain text。"""
    title = DISPLAY_MATH_RE.sub("", title)
    title = INLINE_MATH_RE.sub(lambda m: _plain_math(m.group(1)), title)
    title = EMPTY_PARENS_RE.sub("", title)
    title = re.sub(r"\s+", " ", title).strip(" ·—-")
    return title or fallback


def chapter_title(rel: str) -> str:
    path = BOOK / rel
    fallback = Path(rel).stem
    if path.is_file():
        m = TITLE_RE.search(path.read_text(encoding="utf-8"))
        if m:
            return sanitize_toc_title(m.group(1).strip(), fallback)
    return fallback


def collect_qa_by_volume() -> dict[str, list[str]]:
    """书中所有 qa/*.md，按卷分组（mdBook 仅构建 SUMMARY 列出的页面）。"""
    by_vol: dict[str, list[str]] = {}
    for path in sorted(BOOK.rglob("qa/*.md")):
        rel = path.relative_to(BOOK).as_posix()
        vol = rel.split("/", 1)[0]
        by_vol.setdefault(vol, []).append(rel)
    return by_vol


def qa_list_title(rel: str) -> str:
    title = chapter_title(rel)
    vol = rel.split("/", 1)[0]
    vol_label = SECTION_TITLES.get(vol, vol)
    if rel.endswith("/qa/README.md"):
        return f"{vol_label} · 答疑索引"
    stem = Path(rel).stem
    if stem == "README":
        return f"{vol_label} · 答疑索引"
    return f"{vol_label} · {title}"


def main() -> None:
    lines = ["# Summary", ""]
    lines.append("[中文导读](00-前言/02-中文导读.md)")
    lines.append("")

    current_section: str | None = None
    order = list(build_book.READING_ORDER)
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

    qa_by_vol = collect_qa_by_volume()
    if qa_by_vol:
        lines.append("")
        lines.append("# 答疑")
        for vol in sorted(qa_by_vol.keys()):
            for rel in qa_by_vol[vol]:
                seen.add(rel)
                lines.append(f"- [{qa_list_title(rel)}]({rel})")

    out = BOOK / "SUMMARY.md"
    out.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    qa_count = sum(len(v) for v in qa_by_vol.values())
    print(
        f"WRITE {out.relative_to(REPO)} "
        f"({len(seen) + 1} entries, {qa_count} qa)"
    )


if __name__ == "__main__":
    main()
