#!/usr/bin/env python3
"""Post-process mdBook HTML for GitHub Pages project sites.

1. Replace root index.html with redirect to the real first chapter (avoids ``../``
   resolving outside ``/deepseek-tech-notes/``).
2. Percent-encode non-ASCII path segments in relative href/src (GitHub Pages 400).
"""
from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote, quote

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "mdbook-out"
FIRST_CHAPTER = "00-%E5%89%8D%E8%A8%80/02-%E4%B8%AD%E6%96%87%E5%AF%BC%E8%AF%B2.html"

SKIP_PREFIXES = ("http://", "https://", "//", "#", "mailto:", "javascript:", "data:")
ATTR_RE = re.compile(r'(href|src)="([^"]+)"')


def encode_path(path: str) -> str:
    if not path or path.startswith(SKIP_PREFIXES):
        return path
    frag = ""
    if "#" in path:
        path, frag = path.split("#", 1)
        frag = "#" + frag
    parts = path.split("/")
    encoded = "/".join(
        quote(unquote(part), safe="") if part not in (".", "..", "") else part
        for part in parts
    )
    return encoded + frag


def write_redirect_index() -> None:
    html = f"""<!DOCTYPE html>
<html lang="zh-Hans">
<head>
  <meta charset="utf-8">
  <title>Redirecting…</title>
  <link rel="canonical" href="{FIRST_CHAPTER}">
  <meta http-equiv="refresh" content="0; url={FIRST_CHAPTER}">
  <script>location.replace("{FIRST_CHAPTER}");</script>
</head>
<body>
  <p><a href="{FIRST_CHAPTER}">中文导读</a></p>
</body>
</html>
"""
    (OUT / "index.html").write_text(html, encoding="utf-8")


def patch_html(path: Path) -> int:
    text = path.read_text(encoding="utf-8")

    def repl(m: re.Match[str]) -> str:
        attr, val = m.group(1), m.group(2)
        new = encode_path(val)
        return m.group(0) if new == val else f'{attr}="{new}"'

    patched = ATTR_RE.sub(repl, text)
    if patched != text:
        path.write_text(patched, encoding="utf-8")
    return text.count('href="') + text.count('src="')


def main() -> None:
    if not OUT.is_dir():
        raise SystemExit(f"missing {OUT.relative_to(REPO)} — run mdbook build first")
    write_redirect_index()
    n = 0
    for html in OUT.rglob("*.html"):
        if html.parent == OUT and html.name == "index.html":
            continue
        patch_html(html)
        n += 1
    print(f"OK fix_mdbook_paths: redirect index + encoded paths in {n} html files")


if __name__ == "__main__":
    main()
