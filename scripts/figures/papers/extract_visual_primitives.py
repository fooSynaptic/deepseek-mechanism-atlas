#!/usr/bin/env python3
"""从 Thinking with Visual Primitives PDF 截取 Figure / Table 到 docs/figures/papers/."""
from __future__ import annotations

from pathlib import Path

import pymupdf as fitz

REPO = Path(__file__).resolve().parents[3]
PDF = REPO / "docs" / "papers" / "Thinking_with_Visual_Primitives.pdf"
OUT = REPO / "docs" / "figures" / "papers" / "thinking-with-visual-primitives"

# page 0-based, clip in page coords (612×792 typ.)
JOBS: list[tuple[int, str, fitz.Rect | None]] = [
    # Figure 1 only: bar charts (a)(b) + caption; stop before body text (~y428)
    (1, "fig-1-token-efficiency.png", fitz.Rect(0, 70, 596, 408)),
    # Figure 2: architecture + training pipeline diagram + caption; stop before body (~y349)
    (2, "fig-2-architecture-pipeline.png", fitz.Rect(0, 90, 596, 332)),
    (7, "fig-3-cold-start-counting.png", fitz.Rect(0, 0, 612, 520)),
    (16, "table-1-benchmark.png", fitz.Rect(0, 0, 612, 380)),
]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(PDF)
    mat = fitz.Matrix(2, 2)
    for pno, name, clip in JOBS:
        pix = doc[pno].get_pixmap(matrix=mat, clip=clip, alpha=False)
        pix.save(str(OUT / name))
        print(f"OK {name} ({pix.width}×{pix.height})")
    doc.close()


if __name__ == "__main__":
    main()
