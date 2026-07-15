#!/usr/bin/env python3
"""修复 deepseek-mechanism-atlas 仓内全部本地链接，确保无断链、无项目外引用。"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXT = REPO / "docs/material"
BOOK = REPO / "《ds-技术报告》"

LINK_RE = re.compile(r"(\[[^\]]*\]\()([^)#]+)((?:#[^)]*)?\))")
IMG_RE = re.compile(r'(<img[^>]+src=["\'])([^"\']+)(["\'])')

# 文内批量替换（长模式优先）
BULK_REPLACEMENTS: list[tuple[str, str]] = [
    ("../deepseek-mechanism-atlas/docs/", "../../../docs/"),
    ("../deepseek-mechanism-atlas/《ds-技术报告》/", "../../../《ds-技术报告》/"),
    ("../deepseek-mechanism-atlas/dsa/", "docs/dsa/"),
    ("../reports/deepseek-", "../../../docs/reports/deepseek-"),
    ("../reports/README.md", "../../../docs/reports/README.md"),
    ("../reports/phase", "../../../docs/reports/phase"),
    ("../versions/", "../../../docs/versions/"),
    ("../../../../diagrams/", "../../../diagrams/"),
    ("../../../../../../diagrams/", "../../../diagrams/"),
    ("../../papers/", "../papers/"),
    ("../../../meta/svg-diagram-math.md", "../../../../../meta/svg-diagram-math.md"),
]


def _book_depth(from_md: Path) -> str:
    rel = from_md.relative_to(REPO)
    n = len(rel.parts) - 1
    return "/".join([".."] * n) if n else "."


def resolve_href(from_md: Path, href: str) -> str | None:
    anchor = ""
    if "#" in href:
        href, frag = href.split("#", 1)
        anchor = "#" + frag
    if not href:
        return None

    candidate = (from_md.parent / href).resolve()
    if candidate.exists():
        return href + anchor

    norm = href.replace("\\", "/")

    # 仓名前缀剥离（兼容旧链 deepseek-ai / deepseek-everything）
    for prefix in (
        "deepseek-mechanism-atlas/",
        "deepseek-everything/",
        "deepseek-ai/",
    ):
        if prefix in norm:
            tail = norm.split(prefix, 1)[1]
            target = (REPO / tail).resolve()
            if target.exists():
                return os.path.relpath(target, from_md.parent).replace("\\", "/") + anchor

    # 《ds-技术报告》绝对仓内路径
    if "《ds-技术报告》/" in norm:
        sub = norm.split("《ds-技术报告》/", 1)[1]
        target = (BOOK / sub).resolve()
        if target.exists():
            return os.path.relpath(target, from_md.parent).replace("\\", "/") + anchor

    # 常见 repo 根路径
    for root in ("docs/", "docs/dsa/", "diagrams/", "docs/engram/", "docs/rl/"):
        if norm.startswith(root) or norm.startswith("./" + root):
            clean = norm.removeprefix("./")
            target = (REPO / clean).resolve()
            if target.exists():
                return os.path.relpath(target, from_md.parent).replace("\\", "/") + anchor

    return None


def apply_bulk(text: str) -> str:
    for old, new in BULK_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def fix_book_link_depths(text: str, from_md: Path) -> str:
    """修正过多的 ../《ds-技术报告》/ 层级。"""
    if "《ds-技术报告》/" not in text:
        return text

    def repl(m: re.Match[str]) -> str:
        prefix, href, suffix = m.group(1), m.group(2), m.group(3)
        if "《ds-技术报告》/" not in href:
            return m.group(0)
        sub = href.split("《ds-技术报告》/", 1)[1]
        target = BOOK / sub.split("#")[0]
        if not target.exists():
            return m.group(0)
        anchor = ""
        if "#" in sub:
            _, frag = sub.split("#", 1)
            anchor = "#" + frag
        rel = os.path.relpath(target, from_md.parent).replace("\\", "/")
        return prefix + rel + anchor + suffix

    return LINK_RE.sub(repl, text)


def fix_file(md: Path) -> bool:
    text = md.read_text(encoding="utf-8")
    orig = text
    text = apply_bulk(text)
    text = fix_book_link_depths(text, md)

    def repl_link(m: re.Match[str]) -> str:
        fixed = resolve_href(md, m.group(2))
        return m.group(1) + (fixed if fixed else m.group(2)) + m.group(3)

    def repl_img(m: re.Match[str]) -> str:
        fixed = resolve_href(md, m.group(2))
        return m.group(1) + (fixed if fixed else m.group(2)) + m.group(3)

    text = LINK_RE.sub(repl_link, text)
    text = IMG_RE.sub(repl_img, text)

    if text != orig:
        md.write_text(text, encoding="utf-8")
        return True
    return False


def find_broken() -> list[tuple[Path, str]]:
    broken: list[tuple[Path, str]] = []
    for md in REPO.rglob("*.md"):
        if "《ds-技术报告》" in str(md):
            continue
        for m in LINK_RE.finditer(md.read_text(encoding="utf-8", errors="replace")):
            href = m.group(2).split("#")[0]
            if not href or href.startswith(("http://", "https://", "mailto:")):
                continue
            if href in ("embeddings", "key", "query", "节点文档", "链接") or "..." in href:
                continue
            if not (md.parent / href).resolve().exists():
                broken.append((md, href))
        for m in IMG_RE.finditer(md.read_text(encoding="utf-8", errors="replace")):
            href = m.group(2).split("#")[0]
            if not href or href.startswith(("http://", "https://")):
                continue
            if not (md.parent / href).resolve().exists():
                broken.append((md, href))
    return broken


def find_outside_refs() -> list[tuple[Path, str]]:
    outside: list[tuple[Path, str]] = []
    patterns = ("/mnt/", "/home/", "/root/", "/ai-mlp", "/ivker/", "model_ground/", ".cursor/rules", "gitreps/")
    for md in REPO.rglob("*.md"):
        if "《ds-技术报告》" in str(md):
            continue
        text = md.read_text(encoding="utf-8", errors="replace")
        for m in LINK_RE.finditer(text):
            href = m.group(2)
            if href.startswith(("http://", "https://", "mailto:", "#")):
                continue
            norm = href.replace("\\", "/")
            # docs/material 内部 papers 相对路径合法
            if str(md).startswith(str(EXT)) and (
                norm.startswith("./papers/")
                or norm.startswith("papers/")
                or norm.startswith("../papers/")
            ):
                continue
            if "docs/material/" in norm or norm.startswith("../material/") or "/material/" in str(md):
                if "papers/" in norm and "docs/material" not in norm:
                    target = (md.parent / href.split("#")[0]).resolve()
                    if target.exists() and str(REPO) in str(target):
                        continue
            for p in patterns:
                if p in norm and "docs/material" not in norm:
                    outside.append((md, href))
                    break
    return outside


def main() -> None:
    changed = 0
    for md in REPO.rglob("*.md"):
        if "《ds-技术报告》" in str(md):
            continue
        if fix_file(md):
            changed += 1
    print(f"FIXED {changed} md files")

    broken = find_broken()
    outside = find_outside_refs()
    print(f"BROKEN {len(broken)} OUTSIDE {len(outside)}")
    for md, href in broken[:30]:
        print(f"  {md.relative_to(REPO)} -> {href}")
    for md, href in outside[:15]:
        print(f"  OUT {md.relative_to(REPO)} -> {href}")


if __name__ == "__main__":
    main()
