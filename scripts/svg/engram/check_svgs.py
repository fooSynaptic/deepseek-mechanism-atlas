#!/usr/bin/env python3
"""Validate engram/diagrams/*.svg for XML and basic structure."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "reference"))
from _paths import ENGRAM_DIAGRAMS  # noqa: E402

import xml.etree.ElementTree as ET

def check(path: Path) -> list[str]:
    errs: list[str] = []
    data = path.read_bytes()
    if b"\x00" in data:
        errs.append("contains null bytes")
    try:
        data.decode("utf-8")
    except UnicodeDecodeError as e:
        errs.append(f"utf-8: {e}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        errs.append(f"xml: {e}")
        return errs
    if root.tag != "{http://www.w3.org/2000/svg}svg" and root.tag != "svg":
        errs.append("root is not svg")
    if not path.read_text(encoding="utf-8").lstrip().startswith("<?xml"):
        errs.append("missing <?xml declaration")
    vb = root.attrib.get("viewBox")
    if not vb:
        errs.append("missing viewBox")
    return errs


def main() -> int:
    files = sorted(ENGRAM_DIAGRAMS.glob("*.svg"))
    if not files:
        print("no svg files", file=sys.stderr)
        return 1
    failed = False
    for p in files:
        errs = check(p)
        if errs:
            failed = True
            print(f"FAIL {p.name}: {', '.join(errs)}")
        else:
            print(f"OK   {p.name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
