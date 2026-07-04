#!/usr/bin/env python3
"""从 ESS 论文 PDF 裁剪 Fig.1–9 与 Table 1–2，输出到 docs/figures/ess/paper/。"""
from __future__ import annotations

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / \"svg\"))
from _paths import ESS  # noqa: E402

import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = ESS
PAPER_DIR = ROOT / "paper"
PDF_URL = "https://arxiv.org/pdf/2512.10576.pdf"
PDF_PATH = PAPER_DIR / "ess-2512.10576.pdf"
DPI = 180

# pdfimages 嵌入位图；fig-1 等可改用手动截图（见 install_screenshot.py）
EMBEDDED: dict[str, int] = {}

# (page_1based, (x1, y1, x2, y2)) — 1530×1980 @ 180dpi；下边界避开图注后正文
CROPS: dict[str, tuple[int, tuple[int, int, int, int]]] = {
    "table-1.png": (2, (120, 175, 750, 435)),
    "fig-2.png": (3, (115, 120, 1415, 535)),  # 含图注，不含 §2.2 正文
    "fig-3.png": (4, (115, 75, 1415, 1065)),
    "fig-4.png": (5, (115, 120, 1415, 665)),
    "fig-5.png": (5, (115, 700, 1415, 1280)),
    "fig-6.png": (6, (115, 75, 1415, 500)),
    "fig-7.png": (6, (115, 515, 1415, 920)),
    "fig-8.png": (6, (115, 940, 1415, 1350)),
    "fig-9.png": (7, (115, 75, 1415, 480)),
    "table-2.png": (7, (760, 495, 1415, 1240)),
}


def ensure_pdf() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    if PDF_PATH.exists():
        return
    subprocess.run(
        ["curl", "-fsSL", "-o", str(PDF_PATH), PDF_URL],
        check=True,
    )


def render_pages() -> dict[int, Path]:
    ensure_pdf()
    prefix = PAPER_DIR / "page"
    subprocess.run(
        ["pdftoppm", "-png", "-r", str(DPI), str(PDF_PATH), str(prefix)],
        check=True,
    )
    pages: dict[int, Path] = {}
    for p in range(1, 10):
        path = PAPER_DIR / f"page-{p}.png"
        if not path.exists():
            sys.exit(f"missing rendered page: {path}")
        pages[p] = path
    return pages


def manual_figures() -> set[str]:
    if not (PAPER_DIR / ".manual-figures").exists():
        return set()
    return {
        ln.strip()
        for ln in (PAPER_DIR / ".manual-figures").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    }


def extract_embedded() -> None:
    skip = manual_figures()
    prefix = PAPER_DIR / "pdfimg"
    subprocess.run(
        ["pdfimages", "-png", str(PDF_PATH), str(prefix)],
        check=True,
    )
    for name, idx in EMBEDDED.items():
        if name in skip:
            print("skip (manual)", name)
            continue
        src = PAPER_DIR / f"pdfimg-{idx:03d}.png"
        if not src.exists():
            sys.exit(f"missing embedded image: {src}")
        out = PAPER_DIR / name
        Image.open(src).save(out, optimize=True)
        print("wrote", out.relative_to(ROOT.parent.parent.parent))


def crop_all(pages: dict[int, Path]) -> None:
    skip = manual_figures()
    for name, (page_no, box) in CROPS.items():
        if name in skip:
            print("skip (manual)", name)
            continue
        im = Image.open(pages[page_no])
        out = PAPER_DIR / name
        im.crop(box).save(out, optimize=True)
        print("wrote", out.relative_to(ROOT.parent.parent.parent))


def main() -> None:
    ensure_pdf()
    extract_embedded()
    pages = render_pages()
    crop_all(pages)


if __name__ == "__main__":
    main()
