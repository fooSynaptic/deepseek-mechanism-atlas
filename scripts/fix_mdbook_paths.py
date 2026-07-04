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
MATHJAX_RE = re.compile(r"<!-- MathJax -->\s*<script async src=\"[^\"]+\"></script>")
MATHJAX_BLOCK = """<!-- MathJax: enable $...$ (IDE / VS Code Preview delimiters) -->
        <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
          tex2jax: {
            inlineMath: [['$','$'], ['\\\\(','\\\\)']],
            displayMath: [['$$','$$'], ['\\\\[','\\\\]']],
            processEscapes: true,
            processEnvironments: true,
            skipTags: ['script','noscript','style','textarea','pre','code']
          }
        });
        </script>
        <script async src="https://cdn.jsdelivr.net/npm/mathjax@2.7.9/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>"""


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
        new = resolve_relative(val, page_dir)
        return m.group(0) if new is None else f'{attr}="{new}"'

    text = path.read_text(encoding="utf-8")
    text = ATTR_RE.sub(repl, text)
    text = PATH_TO_ROOT_RE.sub(f'var path_to_root = "{SITE_URL}";', text)
    text = patch_mathjax(text)
    path.write_text(text, encoding="utf-8")


def patch_mathjax(text: str) -> str:
    return MATHJAX_RE.sub(MATHJAX_BLOCK, text)


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
    patch_searchindex()
    print(f"OK fix_mdbook_paths: absolute URLs under {SITE_URL} ({n} chapter html files)")


if __name__ == "__main__":
    main()
