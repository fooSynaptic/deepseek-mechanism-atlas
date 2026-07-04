#!/usr/bin/env python3
"""校验 deepseek-tech-notes 仓内 Markdown 本地链接：零断链、零项目外引用。"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BOOK = REPO / "《ds-技术报告》"
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)#]+)\)|src=[\"']([^\"']+)[\"']")
# 跨仓 / 绝对路径 / 历史迁移残留前缀
OUTSIDE_MARKERS = (
    "/mnt/",
    "/home/",
    "/root/",
    "/ai-mlp",
    "/ivker/",
    "gitreps/",
    "model_ground/",
    ".cursor/rules/",
)
SKIP = {"embeddings", "key", "query", "节点文档", "链接"}


def should_skip_md(md: Path, root: Path) -> bool:
    rel = str(md.relative_to(root)).replace("\\", "/")
    if root == REPO and rel.startswith("《ds-技术报告》/"):
        return True
    if root == BOOK and rel.startswith("09-附录/material/"):
        return True
    prefixes = (
        "docs/material/meta/",
        "scripts/",
        "09-附录/material/meta/",
    )
    return any(rel.startswith(p) for p in prefixes)


def is_outside(href: str, md: Path) -> bool:
    norm = href.replace("\\", "/")
    if norm.startswith(("http://", "https://", "mailto:", "#")):
        return False
    if norm.startswith("/"):
        return True
    if str(md).startswith(str(REPO / "docs/material")):
        if norm.startswith(("./papers/", "../papers/", "papers/")):
            return False
    if "docs/material/" in norm or norm.startswith("../material/"):
        return False
    for m in OUTSIDE_MARKERS:
        if m in norm:
            return True
    if norm.startswith("papers/"):
        target = (md.parent / norm).resolve()
        if str(REPO) not in str(target):
            return True
    return False


def scan(root: Path, label: str) -> tuple[int, int, list, list]:
    broken, outside = [], []
    for md in root.rglob("*.md"):
        if should_skip_md(md, root):
            continue
        for m in LINK_RE.finditer(md.read_text(encoding="utf-8", errors="replace")):
            h = (m.group(1) or m.group(2) or "").split("#")[0]
            if not h or h in SKIP or "..." in h:
                continue
            if h.startswith(("http://", "https://", "mailto:", "#")):
                continue
            if is_outside(h, md):
                outside.append((md.relative_to(REPO), h))
                continue
            if not (md.parent / h).resolve().exists():
                broken.append((md.relative_to(REPO), h))
    print(f"{label}: broken={len(broken)} outside={len(outside)}")
    return len(broken), len(outside), broken, outside


def main() -> int:
    import subprocess

    subprocess.run([sys.executable, str(REPO / "《ds-技术报告》/build_book.py")], check=True)

    b1, o1, br1, ou1 = scan(REPO, "SOURCE")
    b2, o2, br2, ou2 = scan(BOOK, "BOOK")
    for x in br1[:15]:
        print("  B", x[0], "->", x[1])
    for x in ou1[:10]:
        print("  O", x[0], "->", x[1])
    for x in br2[:15]:
        print("  B", x[0], "->", x[1])
    total = b1 + o1 + b2 + o2
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
