#!/usr/bin/env python3
"""迭代修复仓内全部 Markdown 本地链接直至零断链。"""
from __future__ import annotations

import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BOOK = REPO / "《ds-技术报告》"
LINK_RE = re.compile(r"(\[[^\]]*\]\()([^)#]+)((?:#[^)]*)?\))")
IMG_RE = re.compile(r'(<img[^>]+src=["\'])([^"\']+)(["\'])')
SKIP_HREF = {"embeddings", "key", "query", "节点文档", "链接"}


def repo_targets(href: str) -> list[Path]:
    norm = href.replace("\\", "/").split("#")[0]
    if not norm:
        return []
    out: list[Path] = []
    if norm.startswith("/"):
        return out
    candidates = [norm, norm.removeprefix("./")]
    prefixes = [
        "",
        "docs/",
        "docs/material/",
    ]
    for c in candidates:
        out.append(REPO / c)
        for p in prefixes:
            out.append(REPO / p / c.removeprefix(p))
    # 兼容历史链接中的仓名前缀
    for prefix in ("deepseek-tech-notes/", "deepseek-everything/", "deepseek-ai/"):
        if prefix in norm:
            out.append(REPO / norm.split(prefix, 1)[1])
    if "《ds-技术报告》/" in norm:
        out.append(BOOK / norm.split("《ds-技术报告》/", 1)[1])
    return out


def resolve(from_md: Path, href: str) -> str | None:
    anchor = ""
    raw = href
    if "#" in href:
        raw, frag = href.split("#", 1)
        anchor = "#" + frag
    if not raw or "..." in raw or raw in SKIP_HREF:
        return None
    if (from_md.parent / raw).resolve().exists():
        return href
    for t in repo_targets(raw):
        try:
            t = t.resolve()
        except OSError:
            continue
        if t.is_file() or (t.is_dir() and (t / "README.md").is_file()):
            rel = os.path.relpath(t, from_md.parent).replace("\\", "/")
            return rel + anchor
    return None


def patch_file(md: Path) -> int:
    text = md.read_text(encoding="utf-8")
    n = 0

    def repl(m: re.Match[str], kind: str) -> str:
        nonlocal n
        fixed = resolve(md, m.group(2))
        if fixed and fixed != m.group(2):
            n += 1
            return m.group(1) + fixed + m.group(3)
        return m.group(0)

    text2 = LINK_RE.sub(lambda m: repl(m, "link"), text)
    text2 = IMG_RE.sub(lambda m: repl(m, "img"), text2)
    if text2 != text:
        md.write_text(text2, encoding="utf-8")
    return n


def manual_fixes() -> None:
    fixes: list[tuple[str, str, str]] = [
        # path contains, old, new
        (
            "docs/wiki/",
            "../tasks/",
            "../../tasks/",
        ),
        (
            "docs/wiki/",
            "../reports/",
            "../reports/",
        ),
        (
            "docs/material/papers/deepseek-r1/training-pipeline.md",
            "../../../../reports/",
            "../../../../../docs/reports/",
        ),
        (
            "docs/material/papers/engram/engram-series-overview.md",
            "../../../../reports/",
            "../../../../../docs/reports/",
        ),
    ]
    for md in REPO.rglob("*.md"):
        rel = str(md.relative_to(REPO))
        t = md.read_text(encoding="utf-8")
        o = t
        for sub, old, new in fixes:
            if sub in rel:
                t = t.replace(old, new)
        if t != o:
            md.write_text(t, encoding="utf-8")


def broken(md: Path) -> list[str]:
    bad: list[str] = []
    text = md.read_text(encoding="utf-8", errors="replace")
    for pat in (LINK_RE, IMG_RE):
        for m in pat.finditer(text):
            h = m.group(2).split("#")[0]
            if not h or h.startswith(("http://", "https://", "mailto:")) or h in SKIP_HREF or "..." in h:
                continue
            if not (md.parent / h).resolve().exists():
                bad.append(h)
    return bad


def main() -> None:
    manual_fixes()
    for _ in range(5):
        total = 0
        for md in REPO.rglob("*.md"):
            if "《ds-技术报告》" in str(md):
                continue
            total += patch_file(md)
        print(f"pass fixed={total}")
        if total == 0:
            break

    all_broken: list[tuple[str, str]] = []
    for md in REPO.rglob("*.md"):
        if "《ds-技术报告》" in str(md):
            continue
        for h in broken(md):
            all_broken.append((str(md.relative_to(REPO)), h))
    print(f"BROKEN {len(all_broken)}")
    for f, h in all_broken[:40]:
        print(f"  {f} -> {h}")


if __name__ == "__main__":
    main()
