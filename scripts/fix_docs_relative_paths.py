#!/usr/bin/env python3
"""Fix docs/dsa -> relative paths inside docs/ tree."""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"

REPLACEMENTS_BY_DIR: dict[str, list[tuple[str, str]]] = {
    "versions": [
        ("docs/dsa/", "../dsa/"),
        ("docs/rl/", "../rl/"),
        ("docs/engram/", "../engram/"),
        ("../../../docs/reports/", "../reports/"),
        ("../../../docs/versions/", "../versions/"),
    ],
    "reports": [
        ("docs/dsa/", "../dsa/"),
        ("docs/rl/", "../rl/"),
        ("../../docs/engram/", "../engram/"),
        ("../../../docs/versions/", "../versions/"),
        ("../../../docs/reports/", "./"),
    ],
    "wiki": [
        ("docs/dsa/", "dsa/"),
    ],
    "material": [
        ("docs/dsa/", "../../../dsa/"),
        ("docs/engram/", "../../../engram/"),
    ],
    "dsa": [
        ("docs/dsa/", "./"),
    ],
    "rl": [],
    "engram": [],
    "": [  # docs/README.md etc at docs root
        ("docs/dsa/", "dsa/"),
        ("docs/rl/", "rl/"),
        ("docs/engram/", "engram/"),
        ("`dsa/`", "`dsa/`"),  # keep
    ],
}


def rel_bucket(path: Path) -> str:
    rel = path.relative_to(DOCS)
    parts = rel.parts
    if len(parts) == 1:
        return ""
    return parts[0]


def main() -> None:
    n = 0
    for md in DOCS.rglob("*.md"):
        bucket = rel_bucket(md)
        reps = list(REPLACEMENTS_BY_DIR.get(bucket, []))
        if bucket not in ("versions", "reports", "material", "dsa", "wiki", ""):
            reps += REPLACEMENTS_BY_DIR.get("versions", [])
        raw = md.read_text(encoding="utf-8")
        new = raw
        for old, new_s in reps:
            new = new.replace(old, new_s)
        if new != raw:
            md.write_text(new, encoding="utf-8")
            n += 1
            print(md.relative_to(REPO))
    print(f"fixed {n} files")


if __name__ == "__main__":
    main()
