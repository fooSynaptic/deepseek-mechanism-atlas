#!/usr/bin/env python3
"""Remove broken private wiki links; fix docs/dsa and engram relative paths."""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

WIKI_REPLACEMENTS: list[tuple[str, str]] = [
    # scaling / V1 wiki -> in-repo qa or arXiv
    (
        r"\[([^\]]*)\]\(\.\./docs/wiki/缩放定律\.md\)",
        r"[\1](./qa/v1-scaling-law-c-vs-md.md)",
    ),
    (
        r"\[([^\]]*)\]\(\.\./\.\./docs/wiki/缩放定律\.md\)",
        r"[\1](../qa/v1-scaling-law-c-vs-md.md)",
    ),
    (
        r"\[([^\]]*)\]\(\.\./docs/wiki/Scaling-Law选择性应用\.md(?:#[^)]+)?\)",
        r"[\1](./qa/v1-scaling-law-c-vs-md.md)",
    ),
    (
        r"\[([^\]]*)\]\(\.\./\.\./docs/wiki/Scaling-Law选择性应用\.md(?:#[^)]+)?\)",
        r"[\1](../qa/v1-scaling-law-c-vs-md.md)",
    ),
    (
        r"\[([^\]]*)\]\(\.\./docs/wiki/产品训练与Scaling-Law\.md\)",
        r"[\1](./qa/v1-scaling-law-c-vs-md.md)",
    ),
    (
        r"\[([^\]]*)\]\(\.\./\.\./docs/wiki/产品训练与Scaling-Law\.md\)",
        r"[\1](../qa/v1-scaling-law-c-vs-md.md)",
    ),
    (
        r"\[([^\]]*)\]\(\.\./docs/wiki/学习率调度\.md\)",
        r"[\1](./v1.md#训练)",
    ),
    (
        r"\[([^\]]*)\]\(\.\./\.\./docs/wiki/学习率调度\.md\)",
        r"[\1](../v1.md#训练)",
    ),
    (
        r"\[([^\]]*)\]\(\.\./docs/wiki/mla-详解\.md\)",
        r"[\1](./mla-latent-attention.md)",
    ),
    (
        r"\[([^\]]*)\]\(\.\./docs/wiki/deepseek-全系列报告\.md\)",
        r"[\1](../reports/deepseek-version-lineage-20260625.md)",
    ),
    (
        r"\[([^\]]*)\]\(\.\./docs/wiki/reports/source/2412\.19437\.en\.txt\)",
        r"[\1](https://arxiv.org/pdf/2412.19437)",
    ),
    (
        r"\[([^\]]*)\]\(\.\./\.\./docs/wiki/reports/source/2412\.19437\.en\.txt\)",
        r"[\1](https://arxiv.org/pdf/2412.19437)",
    ),
    (
        r"\[([^\]]*)\]\(\.\./docs/wiki/reports/deepseek-llm-v1/source/2401\.02954\.pdf\)",
        r"[\1](https://arxiv.org/pdf/2401.02954)",
    ),
    (
        r"\[([^\]]*)\]\(\.\./docs/wiki/reports/deepseek-llm-v1/technical-highlights\.md\)",
        r"[\1](./v1-bbpe-tokenizer.md)",
    ),
    (
        r"\[([^\]]*)\]\(\.\./docs/wiki/reports/deepseek-llm-v1/translation\.zh\.md\)",
        r"[\1](./v1-technical-report.zh.md)",
    ),
    (
        r"\[([^\]]*)\]\(\.\./docs/wiki/reports/deepseek-llm-v1/README\.md\)",
        r"[\1](./v1.md)",
    ),
    (
        r"\[([^\]]*)\]\(\.\./docs/wiki/reports/deepseek-llm-v1/source/2401\.02954\.en\.txt\)",
        r"[\1](https://arxiv.org/pdf/2401.02954)",
    ),
    (
        r"\[([^\]]*)\]\(\.\./docs/wiki/H20-Scaling-Law实验设计\.md\)",
        r"[\1](./qa/v1-scaling-law-c-vs-md.md)",
    ),
]


def fix_dsa(text: str) -> str:
    return text.replace("../docs/", "../")


def fix_engram_readme(text: str) -> str:
    text = text.replace("../docs/reports/", "../reports/")
    text = text.replace("../docs/material/", "../material/")
    text = text.replace("../《ds-技术报告》/", "../../《ds-技术报告》/")
    if "← 中文导读" not in text[:1200]:
        block = (
            "> [← 中文导读](../README.md) · [← 仓库首页（EN）](../../README.md) · "
            "[← Engram 系列导读](../material/papers/engram/engram-series-overview.md)\n\n"
        )
        text = text.replace("<hr>\n\n", f"<hr>\n\n{block}", 1)
    return text


def fix_engram_material(text: str, path: Path) -> str:
    text = text.replace("../../engram/README.md", "../../../engram/README.md")
    text = text.replace("../../../engram/engram_demo_v1.py", "../../../../engram/engram_demo_v1.py")
    text = text.replace("docs/engram/engram_demo_v1.py", "../../../../engram/engram_demo_v1.py")
    # broken inject_backlinks
    text = re.sub(
        r"> \[← 返回 [^\]]+\]\(\.\./\.\./engram-series-overview\.md\)[^\n]*\n\n",
        "",
        text,
    )
    if "qa" in path.parts and "← 返回" not in text[:500]:
        line = (
            "> [← 返回 Engram 系列导读](../../engram-series-overview.md) · "
            "[答疑目录](./README.md) · [← 中文导读](../../../../README.md) · "
            "[← 仓库首页（EN）](../../../../../README.md)\n\n"
        )
        text = line + text
    return text


def main() -> None:
    n = 0
    for md in REPO.rglob("*.md"):
        if "《ds-技术报告》" in md.parts:
            continue
        raw = md.read_text(encoding="utf-8")
        new = raw
        for pat, repl in WIKI_REPLACEMENTS:
            new = re.sub(pat, repl, new)
        if "docs/dsa" in str(md):
            new = fix_dsa(new)
        if md == REPO / "docs/engram/README.md":
            new = fix_engram_readme(new)
        if "docs/material/papers/engram" in str(md):
            new = fix_engram_material(new, md)
        if md == REPO / "docs/WIKI-INDEX.md":
            new = new.replace("../engram/", "engram/")
        if new != raw:
            md.write_text(new, encoding="utf-8")
            n += 1
            print(md.relative_to(REPO))
    print(f"sanitized {n} md files")


if __name__ == "__main__":
    main()
