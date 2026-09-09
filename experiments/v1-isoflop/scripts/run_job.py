#!/usr/bin/env python3
"""Launch one IsoFLOP job with local torchrun (no SSH)."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("DS_ROOT", Path(__file__).resolve().parents[1]))


def load_job(grid_path: Path, job_id: str) -> dict:
    grid = json.loads(grid_path.read_text())
    job = next((j for j in grid["jobs"] if j["job_id"] == job_id), None)
    if not job:
        raise SystemExit(f"{job_id} not in {grid_path}")
    return job


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--grid", default=str(ROOT / "configs" / "grid.json"))
    ap.add_argument("--nproc", type=int, default=int(os.environ.get("DS_NPROC", "1")))
    ap.add_argument("--data", default=os.environ.get("DS_DATA", str(ROOT / "data")))
    ap.add_argument("--out-root", default=os.environ.get("DS_RESULTS", str(ROOT / "results")))
    ap.add_argument("--python", default=os.environ.get("DS_PY", sys.executable))
    ap.add_argument("--smoke-steps", type=int, default=0)
    ap.add_argument("--no-checkpoint", action="store_true", default=True)
    ap.add_argument("--foreground", action="store_true")
    args = ap.parse_args()

    grid_path = Path(args.grid)
    if not grid_path.is_absolute():
        grid_path = ROOT / grid_path
    job = load_job(grid_path, args.job_id)
    data = Path(args.data)
    outdir = Path(args.out_root) / job["job_id"]
    outdir.mkdir(parents=True, exist_ok=True)
    logdir = Path(os.environ.get("DS_LOGS", ROOT / "logs"))
    logdir.mkdir(parents=True, exist_ok=True)
    log = logdir / f"{job['job_id']}.log"

    extra = []
    if args.smoke_steps:
        extra += ["--smoke-steps", str(args.smoke_steps), "--eval-batches", "1", "--log-every", "1"]
    if args.no_checkpoint:
        extra.append("--no-checkpoint")

    train = [
        str(ROOT / "src" / "train.py"),
        "--job-id",
        job["job_id"],
        "--arch-id",
        job["arch_id"],
        "--c-flops",
        str(job["c_flops"]),
        "--seq-len",
        "4096",
        "--train-bin",
        str(data / "train.bin"),
        "--val-bin",
        str(data / "val.bin"),
        "--train-bytes",
        str(data / "train.bytes.bin"),
        "--val-bytes",
        str(data / "val.bytes.bin"),
        "--out-dir",
        str(outdir),
        "--batch-tokens",
        str(job["batch_tokens"]),
        "--max-lr",
        str(job["max_lr"]),
        "--seed",
        str(job.get("seed", 1)),
        *extra,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if args.nproc <= 1:
        cmd = [args.python, *train]
    else:
        cmd = [
            args.python,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={args.nproc}",
            "--max-restarts=0",
            *train,
        ]
    print("[run_job]", " ".join(cmd), flush=True)
    if args.foreground:
        raise SystemExit(subprocess.call(cmd, env=env))
    logf = log.open("ab")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    (outdir / "pid").write_text(str(proc.pid) + "\n")
    print(f"LAUNCHED pid={proc.pid} log={log} out={outdir}")


if __name__ == "__main__":
    main()
