#!/usr/bin/env python3
"""Serve mdbook-out with GitHub Pages site-url prefix for local smoke tests.

mdbook serve ignores site-url; after fix_mdbook_paths.py links use
/deepseek-mechanism-atlas/... — this server strips that prefix so SVG/HTML both work.
"""
from __future__ import annotations

import mimetypes
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "mdbook-out"
PORT = int(os.environ.get("PORT", "3000"))
PREFIX = "/deepseek-mechanism-atlas"

mimetypes.add_type("image/svg+xml", ".svg")


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(OUT), **kwargs)

    def translate_path(self, path: str) -> str:
        path = unquote(path.split("?", 1)[0])
        if path.startswith(PREFIX + "/"):
            path = path[len(PREFIX) :]
        elif path == PREFIX:
            path = "/"
        return super().translate_path(path)

    def log_message(self, fmt: str, *args) -> None:
        print(fmt % args)


def main() -> None:
    if not OUT.is_dir():
        raise SystemExit(f"missing {OUT} — run: bash scripts/build_pages.sh")
    url = f"http://127.0.0.1:{PORT}{PREFIX}/"
    svg = f"http://127.0.0.1:{PORT}{PREFIX}/05-DSA稀疏注意力/figures/ess-dual-cache.svg"
    print(f"Serving {OUT}")
    print(f"  book: {url}")
    print(f"  svg:  {svg}")
    httpd = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
