#!/usr/bin/env python3
"""将 Markdown 中指向本仓 docs/、dsa/、diagrams/、《ds-技术报告》 的链接改为正确相对路径。"""
from __future__ import annotations

import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BOOK = REPO / "《ds-技术报告》"
HREF_RE = re.compile(r"(\[[^\]]*\]\()([^)#]+)(\)|\")|src=[\"']([^\"']+)[\"']")
SKIP = {"", "#", "embeddings", "key", "query", "节点文档", "链接"}


def resolve_repo(rel: str) -> Path | None:
    stem = rel.split("#")[0]
    if not stem.startswith(("docs/", "docs/dsa/", "diagrams/", "《ds-技术报告》/")):
        return None
    for base in (REPO, BOOK):
        p = base / stem
        if p.exists():
            return p
    return None


def fix_href(md: Path, href: str) -> str | None:
    if href in SKIP or href.startswith(("http://", "https://", "mailto:", "#")) or "..." in href:
        return None
    if (md.parent / href).resolve().exists():
        return None

    anchor = ""
    path_part = href
    if "#" in href:
        path_part, anchor = href.split("#", 1)
        anchor = "#" + anchor

    # 剥前缀 ../ 得到仓内意图路径
    rest = path_part
    while rest.startswith("../"):
        rest = rest[3:]
    target = resolve_repo(rest)
    if target is None:
        return None

    new = os.path.relpath(target, md.parent).replace("\\", "/") + anchor
    if new == href:
        return None
    return new


def process_md(md: Path) -> bool:
    text = md.read_text(encoding="utf-8")
    changed = False

    def repl(m: re.Match) -> str:
        nonlocal changed
        if m.group(2):
            href = m.group(2)
            suffix = m.group(3)
            new = fix_href(md, href)
            if new is None:
                return m.group(0)
            changed = True
            return f"{m.group(1)}{new}{suffix}"
        href = m.group(4)
        new = fix_href(md, href)
        if new is None:
            return m.group(0)
        changed = True
        return f'src="{new}"'

    new_text = HREF_RE.sub(repl, text)
    for bad in (".././",):
        if bad in new_text:
            new_text = new_text.replace(bad, "./")
            changed = True
    if changed:
        md.write_text(new_text, encoding="utf-8")
    return changed


def main() -> None:
    n = 0
    for md in sorted(REPO.rglob("*.md")):
        if BOOK in md.parents and md != BOOK / "README.md":
            # 书中副本由 build_book 生成，跳过源修复
            if str(md).startswith(str(BOOK)):
                continue
        if "《ds-技术报告》" in str(md) and md.parent != REPO:
            continue
        if process_md(md):
            n += 1
            print(md.relative_to(REPO))
    print(f"fixed {n} files")


if __name__ == "__main__":
    main()
