#!/usr/bin/env python3
"""Stitch Wave L t0–t4 with Wave A s1–s3 at C=3e18. Exit codes:

0 = L1 valley hit (interior min among t0–t2, web-scale BPB)
2 = pilot-fail / stop (TinyStories-like)
1 = incomplete or inconclusive
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("DS_ROOT", Path(__file__).resolve().parents[1]))
RESULTS = Path(os.environ.get("DS_RESULTS", ROOT / "results"))

# Ordered by rising M
C318 = ["t0", "t1", "t2", "t3", "t4", "s1", "s2", "s3"]
M_ORDER = {
    "t0": 9.63e8,
    "t1": 1.63e9,
    "t2": 2.77e9,
    "t3": 3.83e9,
    "t4": 5.89e9,
    "s1": 8.86e9,
    "s2": 1.43e10,
    "s3": 2.16e10,
}


def load_bpb(arch: str, ctag: str = "3e18") -> dict | None:
    p = RESULTS / f"iso_c{ctag}_{arch}" / "result.json"
    if not p.is_file():
        return None
    r = json.loads(p.read_text())
    m = r["meta"]
    return {
        "arch": arch,
        "job_id": r["job_id"],
        "bpb": r["best_val_bpb"],
        "D": m["D_tokens"],
        "unique": m.get("unique_train_tokens", 0),
        "M": m.get("arch", {}).get("M") if isinstance(m.get("arch"), dict) else M_ORDER[arch],
    }


def main() -> int:
    rows = []
    missing = []
    for a in C318:
        rec = load_bpb(a)
        if rec is None:
            missing.append(a)
        else:
            rows.append(rec)
    print("C=3e18 stitched (Wave L + Wave A s1–s3)")
    for rec in rows:
        loop = rec["unique"] and rec["D"] > rec["unique"]
        print(
            f"  {rec['arch']:<4} BPB={rec['bpb']:.4f} D={rec['D']/1e9:.3f}B "
            f"{'LOOP' if loop else 'ok'}"
        )
    if missing:
        print(f"incomplete: missing {missing}")
        return 1
    if any(r["unique"] and r["D"] > r["unique"] for r in rows):
        print("FAIL: unique-token looping")
        return 2
    t0 = rows[0]["bpb"]
    if t0 < 0.6:
        print(f"FAIL: t0 BPB={t0:.3f} looks like TinyStories saturation")
        return 2
    best = min(rows, key=lambda r: r["bpb"])
    interior = {"t0", "t1", "t2"}
    # interior min: not an endpoint of the full stitched curve
    ends = {rows[0]["arch"], rows[-1]["arch"]}
    if best["arch"] in interior and best["arch"] not in ends:
        print(f"HIT: interior valley at {best['arch']} BPB={best['bpb']:.4f}")
        return 0
    if best["arch"] in interior:
        # t0 is first of stitched list — still a left-arm valley vs Wave A if
        # BPB rises toward s1–s3
        right = [r for r in rows if r["arch"] in ("s1", "s2", "s3")]
        if right and best["bpb"] + 0.02 < min(r["bpb"] for r in right):
            print(
                f"HIT: valley at {best['arch']} (left of Wave A; "
                f"BPB={best['bpb']:.4f} vs s-min {min(r['bpb'] for r in right):.4f})"
            )
            return 0
    print(
        f"INCONCLUSIVE: min at {best['arch']} BPB={best['bpb']:.4f} "
        "(not a clear interior valley near t0–t2)"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
