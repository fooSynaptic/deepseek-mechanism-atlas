#!/usr/bin/env python3
"""Post-process mdBook HTML for GitHub Pages project sites.

GitHub Pages serves this book at ``/deepseek-tech-notes/``.  mdBook emits
relative ``../`` links that escape the project prefix on root/index pages, and
unencoded CJK path segments return 400.  Rewrite internal links to absolute,
percent-encoded URLs under site-url and fix search index paths to match.
"""
from __future__ import annotations

import json
import re
from pathlib import Path, PurePosixPath
from urllib.parse import quote, unquote

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "mdbook-out"
BOOK_TOML = REPO / "book.toml"

SKIP_PREFIXES = ("http://", "https://", "//", "mailto:", "javascript:", "data:")
ATTR_RE = re.compile(r'(href|src)="([^"]+)"')
PATH_TO_ROOT_RE = re.compile(r'var path_to_root = "[^"]*";')
# 图示详情等静态资源保持相对路径：Pages 上相对 chapter 可解析，且便于本地 serve
STATIC_ASSET_SUFFIXES = {
    ".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico",
    ".css", ".js", ".woff", ".woff2", ".ttf", ".json",
}


def is_static_asset(href: str) -> bool:
    path = href.split("#", 1)[0].split("?", 1)[0]
    return PurePosixPath(unquote(path)).suffix.lower() in STATIC_ASSET_SUFFIXES
MATHJAX_RE = re.compile(
    r"<!-- MathJax[^>]* -->[\s\S]*?"
    r"<script async src=\"[^\"]*(?:MathJax|mathjax)[^\"]*\"></script>\s*",
    re.IGNORECASE,
)
KATEX_HEAD = """<!-- KaTeX: $...$ / $$...$$ (IDE / VS Code Preview delimiters) -->
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
"""


def read_site_url() -> str:
    text = BOOK_TOML.read_text(encoding="utf-8")
    m = re.search(r'^site-url\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if not m:
        raise SystemExit("book.toml missing output.html site-url")
    url = m.group(1)
    return url if url.endswith("/") else url + "/"


SITE_URL = read_site_url()


def encode_segment(part: str) -> str:
    if part in (".", "..", ""):
        return part
    return quote(unquote(part), safe="")


def encode_book_rel(rel: str) -> str:
    frag = ""
    if "#" in rel:
        rel, frag = rel.split("#", 1)
        frag = "#" + frag
    encoded = "/".join(encode_segment(p) for p in rel.split("/") if p)
    return encoded + frag


def site_href(rel: str) -> str:
    return SITE_URL + encode_book_rel(rel)


def resolve_relative(href: str, page_dir: PurePosixPath) -> str | None:
    if not href or href.startswith(SKIP_PREFIXES):
        return None
    if href.startswith(SITE_URL):
        return href
    frag = ""
    if "#" in href:
        href, frag = href.split("#", 1)
        frag = "#" + frag
    if href.startswith("/"):
        rel = href.lstrip("/")
    else:
        rel = PurePosixPath(page_dir, href).as_posix()
    parts: list[str] = []
    for part in rel.split("/"):
        if part == "..":
            if parts:
                parts.pop()
        elif part not in (".", ""):
            parts.append(part)
    if not parts:
        return None
    return site_href("/".join(parts)) + frag


def patch_html(path: Path) -> None:
    page_dir = PurePosixPath(path.relative_to(OUT).parent.as_posix())
    if page_dir.parts == (".",):
        page_dir = PurePosixPath()

    def repl(m: re.Match[str]) -> str:
        attr, val = m.group(1), m.group(2)
        if is_static_asset(val):
            return m.group(0)
        new = resolve_relative(val, page_dir)
        return m.group(0) if new is None else f'{attr}="{new}"'

    text = path.read_text(encoding="utf-8")
    text = ATTR_RE.sub(repl, text)
    text = PATH_TO_ROOT_RE.sub(f'var path_to_root = "{SITE_URL}";', text)
    text = patch_mathjax(text)
    path.write_text(text, encoding="utf-8")


def patch_mathjax(text: str) -> str:
    if "<!-- KaTeX:" in text or "katex.min.js" in text:
        return text
    if MATHJAX_RE.search(text):
        return MATHJAX_RE.sub(KATEX_HEAD, text, count=1)
    return text.replace("</head>", KATEX_HEAD + "    </head>", 1)


def write_redirect_index() -> None:
    target = site_href("00-前言/02-中文导读.html")
    html = f"""<!DOCTYPE html>
<html lang="zh-Hans">
<head>
  <meta charset="utf-8">
  <title>Redirecting…</title>
  <link rel="canonical" href="{target}">
  <meta http-equiv="refresh" content="0; url={target}">
  <script>location.replace("{target}");</script>
</head>
<body>
  <p><a href="{target}">中文导读</a></p>
</body>
</html>
"""
    (OUT / "index.html").write_text(html, encoding="utf-8")


def write_removed_page_redirects() -> None:
    """Stub pages dropped from SUMMARY — old bookmarks land on the replacement chapter."""
    removed: dict[str, str] = {
        "00-前言/01-知识库导读.html": "02-中文导读.html",
    }
    for src_rel, target in removed.items():
        out = OUT / src_rel
        out.parent.mkdir(parents=True, exist_ok=True)
        title = "已移除 · 跳转中文导读"
        html = f"""<!DOCTYPE html>
<html lang="zh-Hans">
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <meta http-equiv="refresh" content="0; url={target}">
  <script>location.replace("{target}");</script>
</head>
<body>
  <p>「知识库导读」已并入 <a href="{target}">中文导读</a>。</p>
</body>
</html>
"""
        out.write_text(html, encoding="utf-8")


def patch_searchindex() -> None:
    path = OUT / "searchindex.json"
    if not path.is_file():
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    data["doc_urls"] = [encode_book_rel(u) for u in data["doc_urls"]]
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    js = OUT / "searchindex.js"
    if js.is_file():
        js.write_text(
            "Object.assign(window.search, " + json.dumps(data, ensure_ascii=False) + ");",
            encoding="utf-8",
        )


def main() -> None:
    if not OUT.is_dir():
        raise SystemExit(f"missing {OUT.relative_to(REPO)} — run mdbook build first")
    n = 0
    for html in OUT.rglob("*.html"):
        if html.parent == OUT and html.name == "index.html":
            continue
        patch_html(html)
        n += 1
    write_redirect_index()
    write_removed_page_redirects()
    patch_searchindex()
    print(f"OK fix_mdbook_paths: absolute URLs under {SITE_URL} ({n} chapter html files)")


if __name__ == "__main__":
    main()
