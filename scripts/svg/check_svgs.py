#!/usr/bin/env python3
"""Validate deepseek-ai SVG diagrams + Markdown embed refs + layout."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _paths import DSA_DIAGRAMS, DIAGRAMS, ENGRAM_DIAGRAMS, REPO  # noqa: E402
from svg_validate import check_markdown_svg_refs, validate_svg  # noqa: E402


def main() -> int:
    failed = False
    svg_dirs = [DIAGRAMS, DSA_DIAGRAMS, ENGRAM_DIAGRAMS]
    checked: set[Path] = set()
    for d in svg_dirs:
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.svg")):
            if p in checked:
                continue
            checked.add(p)
            errs = validate_svg(p)
            if errs:
                failed = True
                print(f"FAIL {p.relative_to(REPO)}: {', '.join(errs)}")
            else:
                print(f"OK   {p.relative_to(REPO)}")

    md_errs = check_markdown_svg_refs(REPO)
    if md_errs:
        failed = True
        print("FAIL markdown svg embeds:")
        for e in md_errs:
            print(f"  - {e}")
    else:
        print("OK   markdown svg embeds")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
