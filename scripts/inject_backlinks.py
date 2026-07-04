#!/usr/bin/env python3
"""Inject bidirectional back-links into qa/*.md and topic index pages."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

HOMEPAGE_ZH = "[← 中文导读](../README.md)"
HOMEPAGE_EN = "[← 仓库首页（EN）](../../README.md)"
HOMEPAGE_EN_FROM_REPORTS = "[← 仓库首页（EN）](../../README.md)"

# qa file -> parent article path (relative from qa file)
QA_PARENTS: dict[str, str] = {
    "docs/versions/qa/mtp-fusion-scheme.md": "../dspark-speculative-decoding.md",
    "docs/versions/qa/v4-indexer-kv.md": "../csa-hca-mixed-attention.md#v4-mixed-attention",
    "docs/versions/qa/v1-scaling-law-c-vs-md.md": "../v1.md",
}

DSA_BACK = (
    "> [← 中文导读](../README.md) · [← 仓库首页（EN）](../../README.md) · "
    "[← DSA 系列导读](./README.md) · [← 演进总览 §3.6](../reports/deepseek-version-lineage-20260625.md#36-deepseek-v32--v32-exp)"
)
RL_BACK = (
    "> [← 中文导读](../README.md) · [← 仓库首页（EN）](../../README.md) · "
    "[← RL 笔记索引](./README.md) · [← RLVR](../versions/rlvr.md)"
)
ENGRAM_BACK = (
    "> [← 中文导读](../README.md) · [← 仓库首页（EN）](../../README.md) · "
    "[← Engram 官方 README](./README.md) · [← Engram 系列导读](../material/papers/engram/engram-series-overview.md)"
)

TOPIC_HEADERS: dict[Path, str] = {
    REPO / "docs/dsa/dsa-logic.md": DSA_BACK,
    REPO / "docs/dsa/lightning-indexer.md": DSA_BACK,
    REPO / "docs/dsa/index-share-logic.md": DSA_BACK,
    REPO / "docs/rl/optimize.md": RL_BACK,
}


def has_back_link(text: str) -> bool:
    return "← 中文导读" in text or "← 返回" in text


def inject_after_title(path: Path, block: str) -> bool:
    raw = path.read_text(encoding="utf-8")
    if has_back_link(raw[:800]):
        return False
    m = re.search(r"^(# .+\n)\n?", raw)
    if not m:
        return False
    insert_at = m.end()
    new = raw[:insert_at] + block + "\n\n" + raw[insert_at:]
    path.write_text(new, encoding="utf-8")
    return True


def inject_qa(path: Path, parent_rel: str) -> bool:
    raw = path.read_text(encoding="utf-8")
    if has_back_link(raw[:400]):
        return False
    parent_name = parent_rel.split("/")[-1].split("#")[0]
    back = f"[← 返回 {parent_name}]({parent_rel})"
    idx = "./README.md" if (path.parent / "README.md").exists() else None
    parts = [back]
    if idx:
        parts.append(f"[答疑目录]({idx})")
    parts.extend([HOMEPAGE_ZH.replace("../README", "../../README"), HOMEPAGE_EN])
    line = "> " + " · ".join(parts) + "\n\n"
    new = line + raw
    path.write_text(new, encoding="utf-8")
    return True


def main() -> None:
    changed = 0
    for rel, parent in QA_PARENTS.items():
        p = REPO / rel
        if p.exists() and inject_qa(p, parent):
            changed += 1
            print(f"qa: {rel}")

    # engram material qa
    engram_qa = REPO / "docs/material/papers/engram/qa"
    if engram_qa.is_dir():
        overview = "../../engram-series-overview.md"
        for qa in engram_qa.glob("*.md"):
            if qa.name == "README.md":
                continue
            if inject_qa(qa, overview):
                changed += 1
                print(f"qa: {qa.relative_to(REPO)}")

    for path, block in TOPIC_HEADERS.items():
        if path.exists() and inject_after_title(path, block):
            changed += 1
            print(f"topic: {path.relative_to(REPO)}")

    print(f"backlinks: {changed} files updated")


if __name__ == "__main__":
    main()
