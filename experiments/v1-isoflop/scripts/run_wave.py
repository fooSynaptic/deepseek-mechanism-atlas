#!/usr/bin/env python3
"""Sequential local queue: skip finished result.json, refuse looping D."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("DS_ROOT", Path(__file__).resolve().parents[1]))


def unique_train(data: Path) -> int:
    p = data / "train.bin"
    if not p.exists():
        raise SystemExit(f"missing {p}; run scripts/prepare_data.sh first")
    return p.stat().st_size // 2  # uint16


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=str(ROOT / "configs" / "grid.json"))
    ap.add_argument("--nproc", type=int, default=int(os.environ.get("DS_NPROC", "1")))
    ap.add_argument("--data", default=os.environ.get("DS_DATA", str(ROOT / "data")))
    ap.add_argument("--max-jobs", type=int, default=0, help="0 = whole grid")
    ap.add_argument("--smoke-steps", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    grid_path = Path(args.grid)
    if not grid_path.is_absolute():
        grid_path = ROOT / grid_path
    grid = json.loads(grid_path.read_text())
    data = Path(args.data)
    results = Path(os.environ.get("DS_RESULTS", ROOT / "results"))
    n_train = unique_train(data)
    launched = 0
    for job in grid["jobs"]:
        jid = job["job_id"]
        out = results / jid
        if (out / "result.json").exists():
            print(f"[skip] {jid} has result.json")
            continue
        if int(job["D_tokens"]) > n_train and args.smoke_steps == 0:
            print(f"[wait-data] {jid} D={job['D_tokens']} > unique={n_train}")
            continue
        print(f"[queue] {jid} C={job['c_flops']:.0e} D={job['D_tokens']}")
        if args.dry_run:
            launched += 1
            if args.max_jobs and launched >= args.max_jobs:
                break
            continue
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "run_job.py"),
            "--job-id",
            jid,
            "--grid",
            str(grid_path),
            "--nproc",
            str(args.nproc),
            "--data",
            str(data),
            "--foreground",
        ]
        if args.smoke_steps:
            cmd += ["--smoke-steps", str(args.smoke_steps)]
        rc = subprocess.call(cmd)
        if rc != 0:
            raise SystemExit(f"{jid} failed rc={rc}")
        launched += 1
        if args.max_jobs and launched >= args.max_jobs:
            break
        time.sleep(1)
    print(f"[wave] launched={launched}")


if __name__ == "__main__":
    main()
