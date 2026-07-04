#!/usr/bin/env python3
"""Export repo SVGs to png/ (CJK-safe; source diagrams only, no book copies)."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PNG_ROOT = REPO / "png"
SKIP_PARTS = {
    ".git",
    "png",
    "__pycache__",
    "node_modules",
    "《ds-技术报告》",  # book mirrors — skip duplicates
}
# Canonical SVG roots (avoid duplicate exports from docs/figures copies)
CANONICAL_ROOTS = (
    REPO / "diagrams",
    REPO / "docs" / "dsa" / "diagrams",
    REPO / "docs" / "figures",
    REPO / "docs" / "versions" / "figures",
    REPO / "docs" / "material" / "papers" / "engram" / "diagrams",
)


def _in_canonical(svg: Path) -> bool:
    for root in CANONICAL_ROOTS:
        try:
            svg.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def main() -> None:
    sys.path.insert(0, str(REPO / "scripts" / "svg"))
    from cjk_png import svg_to_png  # noqa: E402

    if PNG_ROOT.exists():
        for p in PNG_ROOT.rglob("*.png"):
            p.unlink()
    else:
        PNG_ROOT.mkdir()

    count = 0
    for svg in sorted(REPO.rglob("*.svg")):
        if any(part in SKIP_PARTS for part in svg.parts):
            continue
        if not _in_canonical(svg):
            continue
        rel = svg.relative_to(REPO)
        out = PNG_ROOT / rel.with_suffix(".png")
        svg_to_png(svg, out)
        count += 1
    print(f"exported {count} PNGs (CJK) -> {PNG_ROOT}")


if __name__ == "__main__":
    main()
