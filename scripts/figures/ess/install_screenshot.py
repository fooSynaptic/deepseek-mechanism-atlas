#!/usr/bin/env python3
"""把用户截图安装到 ess-paper-highlights 引用的 paper/ 目录。

用法：
  # 单张：命名好 fig-N 或 table-N
  python3 install_screenshot.py fig-1 ~/Downloads/截屏.png

  # 批量：先把文件放进 paper/screenshots/，再
  python3 install_screenshot.py --inbox

截图建议：只含「图 + 原论文图注」，不要页眉/正文。已登记 .manual-figures 的文件不会被 extract_paper_figures.py 覆盖。
"""
from __future__ import annotations

from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / \"svg\"))
from _paths import ESS  # noqa: E402

import argparse
import shutil
import sys
from pathlib import Path

ROOT = ESS
PAPER = ROOT / "paper"
INBOX = PAPER / "screenshots"
MANUAL = PAPER / ".manual-figures"

ALLOWED = {f"fig-{i}.png" for i in range(1, 10)} | {"table-1.png", "table-2.png"}


def register_manual(name: str) -> None:
    names = set()
    if MANUAL.exists():
        names = {ln.strip() for ln in MANUAL.read_text(encoding="utf-8").splitlines() if ln.strip()}
    if name not in names:
        names.add(name)
        MANUAL.write_text("\n".join(sorted(names)) + "\n", encoding="utf-8")


def install_one(src: Path, dest_name: str) -> None:
    if dest_name not in ALLOWED:
        sys.exit(f"unknown dest name: {dest_name} (use fig-1..fig-9, table-1, table-2)")
    if not src.is_file():
        sys.exit(f"not found: {src}")
    PAPER.mkdir(parents=True, exist_ok=True)
    dest = PAPER / dest_name
    shutil.copy2(src, dest)
    register_manual(dest_name)
    print(f"OK  {src} -> {dest}")


def install_inbox() -> None:
    if not INBOX.is_dir():
        sys.exit(f"inbox missing: {INBOX}")
    files = sorted(INBOX.glob("*.png"))
    if not files:
        sys.exit(f"no png in {INBOX}")
    for p in files:
        install_one(p, p.name)


def main() -> None:
    ap = argparse.ArgumentParser(description="Install ESS paper figure screenshots")
    ap.add_argument("dest", nargs="?", help="e.g. fig-1")
    ap.add_argument("src", nargs="?", help="source image path")
    ap.add_argument("--inbox", action="store_true", help="install all paper/screenshots/*.png")
    args = ap.parse_args()
    if args.inbox:
        install_inbox()
        return
    if not args.dest or not args.src:
        ap.print_help()
        sys.exit(1)
    install_one(Path(args.src).expanduser().resolve(), args.dest)


if __name__ == "__main__":
    main()
