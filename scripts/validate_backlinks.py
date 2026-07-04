#!/usr/bin/env python3
"""校验 Markdown 双向引用：顶栏回链 + 非 Hub 文档间的成对反向链接。"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote

REPO = Path(__file__).resolve().parents[1]
LINK_RE = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
SKIP_HREF = {"embeddings", "key", "query", "节点文档", "链接"}
SKIP_EXT = {".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".pdf", ".txt", ".py", ".otf", ".drawio"}

# 不要求成对反向的源/目标（索引、总览、维护页）
HUB_RELS = frozenset(
    {
        "docs/README.md",
        "docs/WIKI-INDEX.md",
        "docs/versions/README.md",
        "docs/reports/README.md",
        "docs/reports/deepseek-version-lineage-20260625.md",
        "docs/reports/deepseek-algorithm-line.md",
        "docs/reports/deepseek-infra-line.md",
        "docs/reports/deepseek-moe-line.md",
        "docs/reports/deepseek-doc-series-audit-20260627.md",
        "docs/material/README.md",
        "docs/dsa/README.md",
        "docs/rl/README.md",
        "docs/engram/README.md",
    }
)
SKIP_PREFIX = (
    "scripts/",
    "docs/material/meta/",
    "docs/figures/ess/paper/",
    "docs/material/papers/engram/diagrams/reference/",
)
# 顶栏回链豁免（自身即入口）
SKIP_BACK_BAR = frozenset({"docs/README.md"})


def rel(p: Path) -> str:
    return str(p.relative_to(REPO)).replace("\\", "/")


def resolve(from_md: Path, href: str) -> Path | None:
    raw = unquote(href.split("#")[0].strip())
    if not raw or raw.startswith(("http://", "https://", "mailto:")):
        return None
    if raw in SKIP_HREF or "..." in raw:
        return None
    if any(raw.lower().endswith(ext) for ext in SKIP_EXT):
        return None
    if "." in Path(raw).name and not raw.endswith(".md"):
        return None
    target = (from_md.parent / raw).resolve()
    if target.is_dir():
        target = target / "README.md"
    if target.suffix != ".md":
        target = Path(str(target) + ".md")
    if target.is_file() and str(REPO) in str(target):
        return target
    return None


def extract_md_links(md: Path) -> set[str]:
    out: set[str] = set()
    for m in LINK_RE.finditer(md.read_text(encoding="utf-8", errors="replace")):
        t = resolve(md, m.group(1))
        if t:
            out.add(rel(t))
    return out


def has_hub_back_bar(text: str) -> bool:
    head = text[:1500]
    if "←" not in head:
        return False
    markers = (
        "← 算法线导读",
        "← 基础设施线导读",
        "← MoE 线导读",
        "← 演进总览",
        "← 全系列演进总览",
        "← 中文导读",
        "← 返回",
        "← DSA",
        "← RL",
        "← Engram",
        "← 报告目录",
        "← 梗概",
        "← 系列上下文",
        "← 开发索引",
    )
    return any(m in head for m in markers)


def links_to(text: str, src_rel: str, from_md: Path) -> bool:
    for m in LINK_RE.finditer(text):
        t = resolve(from_md, m.group(1))
        if t and rel(t) == src_rel:
            return True
    return False


def is_hub(r: str) -> bool:
    if r in HUB_RELS:
        return True
    if r.endswith("/README.md") and "/qa/" in r:
        return True
    return False


def should_scan(r: str) -> bool:
    if not r.startswith("docs/") or not r.endswith(".md"):
        return False
    if any(r.startswith(p) for p in SKIP_PREFIX):
        return False
    return True


def check_back_bars() -> list[str]:
    missing: list[str] = []
    for md in sorted(REPO.glob("docs/**/*.md")):
        r = rel(md)
        if not should_scan(r) or r in SKIP_BACK_BAR:
            continue
        if not has_hub_back_bar(md.read_text(encoding="utf-8", errors="replace")):
            missing.append(r)
    return missing


def check_pairwise() -> list[tuple[str, str]]:
    edges: list[tuple[str, str]] = []
    files = {rel(p): p for p in REPO.glob("docs/**/*.md") if should_scan(rel(p))}
    for sr, sm in files.items():
        for tr in extract_md_links(sm):
            if tr == sr or tr not in files:
                continue
            edges.append((sr, tr))

    missing: list[tuple[str, str]] = []
    for src, dst in edges:
        if is_hub(src) or is_hub(dst):
            continue
        dst_md = files[dst]
        body = dst_md.read_text(encoding="utf-8", errors="replace")
        if links_to(body, src, dst_md):
            continue
        if has_hub_back_bar(body):
            continue
        missing.append((src, dst))
    return missing


def main() -> int:
    bar = check_back_bars()
    pair = check_pairwise()
    print(f"back_bar missing: {len(bar)}")
    for r in bar:
        print(f"  BAR  {r}")
    print(f"pairwise reverse missing: {len(pair)}")
    for s, d in pair[:30]:
        print(f"  PAIR {s}\n       -> {d}")
    if len(pair) > 30:
        print(f"  ... and {len(pair) - 30} more")
    return 1 if bar or pair else 0


if __name__ == "__main__":
    raise SystemExit(main())
