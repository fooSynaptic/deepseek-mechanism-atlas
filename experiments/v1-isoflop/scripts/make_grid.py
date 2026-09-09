#!/usr/bin/env python3
"""IsoFLOP grids.

Default: Wave A (s1–s6 × 3 C) → configs/grid.json (do not overwrite after Wave A).
--wave l1: t0–t4 × C=3e18 → configs/grid_wave_l.json
--wave l2: t0–t4 × C=1e19 + t2–t4 × C=3e19 → configs/grid_wave_l2.json
--wave l3: t1 then t0 × C=3e19 → configs/grid_wave_l3.json
  (t1 first: unlocks at ~18.4B unique; t0 needs ~31.2B)
--wave l4: left-of-t0 (u0–u3) on the frozen ~40B unique set, no new fetch.
  Drop any (C, width) whose D would loop. → configs/grid_wave_l4.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flops import (  # noqa: E402
    N_GPUS,
    WAVE_A_ARCHS,
    WAVE_L_ARCHS,
    WAVE_L4_ARCHS,
    formula1_batch_tokens,
    formula1_lr,
    make_arch,
    tokens_for_compute,
)

SEQ = 4096
VOCAB = 50257
CS_A = [3e18, 1e19, 3e19]
# Frozen unique-token ceiling used for L4. L4 must not fetch more data.
L4_UNIQUE_TOKENS = 39_995_303_545


def job_row(c: float, arch_id: str) -> dict:
    arch = make_arch(arch_id, seq_len=SEQ, n_vocab=VOCAB)
    d = tokens_for_compute(c, arch.M)
    ctag = f"{c:.0e}".replace("+0", "").replace("+", "")
    return {
        "job_id": f"iso_c{ctag}_{arch_id}",
        "family": "isoflop",
        "arch_id": arch_id,
        "c_flops": c,
        "D_tokens": d,
        "batch_tokens": formula1_batch_tokens(c),
        "max_lr": formula1_lr(c),
        "seed": 1,
        "n_gpus": N_GPUS[arch_id],
        **arch.scale_table(),
    }


def write_grid(jobs: list[dict], dest: Path) -> None:
    out = {"seq_len": SEQ, "n_vocab": VOCAB, "n_jobs": len(jobs), "jobs": jobs}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {dest} n={len(jobs)}")
    print(f"{'job_id':<24} {'C':>8} {'N1':>8} {'D':>12} gpus")
    for j in jobs:
        print(
            f"{j['job_id']:<24} {j['c_flops']:.0e} {j['N1']/1e9:6.2f}B "
            f"{j['D_tokens']:>12d} {j['n_gpus']}"
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wave", choices=["a", "l1", "l2", "l3", "l4"], default="a")
    args = p.parse_args()
    if args.wave == "a":
        jobs = [job_row(c, a) for c in CS_A for a in WAVE_A_ARCHS]
        write_grid(jobs, ROOT / "configs" / "grid.json")
    elif args.wave == "l1":
        jobs = [job_row(3e18, a) for a in WAVE_L_ARCHS]
        write_grid(jobs, ROOT / "configs" / "grid_wave_l.json")
    elif args.wave == "l2":
        jobs = [job_row(1e19, a) for a in WAVE_L_ARCHS]
        jobs += [job_row(3e19, a) for a in ("t2", "t3", "t4")]
        write_grid(jobs, ROOT / "configs" / "grid_wave_l2.json")
    elif args.wave == "l3":
        # t1 before t0 so the queue can start once unique ≥ ~18.4B
        jobs = [job_row(3e19, a) for a in ("t1", "t0")]
        write_grid(jobs, ROOT / "configs" / "grid_wave_l3.json")
    else:
        # Smaller C first (left-arm readout), then 1e19, then the one 3e19 point.
        jobs = []
        for c in CS_A:
            for a in WAVE_L4_ARCHS:
                row = job_row(c, a)
                if int(row["D_tokens"]) <= L4_UNIQUE_TOKENS:
                    jobs.append(row)
        write_grid(jobs, ROOT / "configs" / "grid_wave_l4.json")


if __name__ == "__main__":
    main()
