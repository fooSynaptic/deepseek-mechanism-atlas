#!/usr/bin/env python3
"""Crop key figures from Engram-Nine PDF (arXiv:2601.16531) for wiki embedding."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[3]  # engram/
PDF = ROOT / "src" / "2601.16531.pdf"
OUT = Path(__file__).resolve().parent

# page, filename, (left, upper, right, lower) at 200 DPI
CROPS: list[tuple[int, str, tuple[int, int, int, int]]] = [
    (4, "fig1-two-tier-architecture", (90, 320, 1620, 820)),
    (7, "table3-val-loss-summary", (90, 250, 1620, 1180)),
    (7, "table4-flip-timing", (90, 1180, 1620, 1580)),
    (8, "fig2-hot-cold-flip", (90, 55, 1620, 780)),
    (8, "table5-nhot-ablation", (90, 1180, 1620, 1580)),
    (9, "fig3-gate-evolution", (90, 55, 1620, 1320)),
    (10, "table6-alpha-bucket", (90, 60, 1620, 520)),
    (10, "fig4-gate-mismatch", (90, 540, 1620, 1080)),
    (11, "fig5-layer-alpha-reversal", (90, 60, 1620, 620)),
]


def main() -> None:
    if not PDF.is_file():
        raise SystemExit(f"PDF not found: {PDF}")
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        prefix = Path(tmp) / "page"
        subprocess.run(
            ["pdftoppm", "-png", "-r", "200", str(PDF), str(prefix)],
            check=True,
        )
        for page, name, box in CROPS:
            src = Path(tmp) / f"page-{page:02d}.png"
            im = Image.open(src)
            im.crop(box).save(OUT / f"{name}.png", optimize=True)
            print(f"wrote {name}.png")


if __name__ == "__main__":
    main()
