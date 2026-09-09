#!/usr/bin/env python3
"""Launch 20 1-GPU jobs over ssh from configs/grid.json."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("DS_ROOT", Path(__file__).resolve().parents[1]))
NODES = os.environ.get("DS_NODES", "").strip()
PY = os.environ.get("DS_REMOTE_PY", str(ROOT / ".venv" / "bin" / "python"))
DATA = Path(os.environ.get("DS_DATA", ROOT / "data"))
RESULTS = Path(os.environ.get("DS_RESULTS", ROOT / "results"))
LOGS = Path(os.environ.get("DS_LOGS", ROOT / "logs"))
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
HF_HOME = os.environ.get("HF_HOME", str(ROOT / ".hf"))


def ssh(host: str, script: str) -> str:
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host, "bash", "-s"],
        input=script,
        text=True,
        capture_output=True,
    )
    if r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"ssh {host} failed: {r.returncode}")
    return r.stdout


def slots():
    out = []
    for spec in NODES.split():
        host, ngpu = spec.split(":")
        for g in range(int(ngpu)):
            out.append((host, g))
    return out


def main():
    if not NODES:
        raise SystemExit(
            "Set DS_NODES='host:ngpus ...' for this SSH launcher, "
            "or use scripts/run_wave.py for a local sequential run."
        )
    local_root = Path(__file__).resolve().parents[1]
    grid_path = Path(os.environ.get("GRID", local_root / "configs" / "grid.json"))
    if not grid_path.exists():
        grid_path = ROOT / "configs" / "grid.json"
    grid = json.loads(grid_path.read_text())
    jobs = grid["jobs"]
    sl = slots()
    if len(jobs) != len(sl):
        raise SystemExit(f"jobs={len(jobs)} slots={len(sl)}")
    train_bin = os.environ.get("TRAIN_BIN", str(DATA / "train.bin"))
    val_bin = os.environ.get("VAL_BIN", str(DATA / "val.bin"))
    train_bytes = os.environ.get("TRAIN_BYTES", str(DATA / "train.bytes.bin"))
    val_bytes = os.environ.get("VAL_BYTES", str(DATA / "val.bytes.bin"))
    head = sl[0][0]
    chk = ssh(head, f"test -f {train_bin} && test -f {val_bin} && echo OK")
    if "OK" not in chk:
        raise SystemExit(f"missing corpus on {head}: {train_bin}")

    plan = []
    for job, (host, gpu) in zip(jobs, sl):
        plan.append(
            {
                "job_id": job["job_id"],
                "host": host,
                "gpu": gpu,
                "arch_id": job["arch_id"],
                "c_flops": job["c_flops"],
                "batch_tokens": job["batch_tokens"],
                "max_lr": job["max_lr"],
                "seed": job.get("seed", 1),
            }
        )
    ssh(head, f"mkdir -p {LOGS} {RESULTS}")
    plan_json = json.dumps(plan, indent=2)
    ssh(head, f"cat > {LOGS}/launch_plan.json <<'PLAN'\n{plan_json}\nPLAN\n")
    print(f"{'job':<22} {'host':<12} gpu")
    for p in plan:
        print(f"{p['job_id']:<22} {p['host']:<12} {p['gpu']}")
        outdir = RESULTS / p["job_id"]
        log = LOGS / f"{p['job_id']}.log"
        script = f"""
set -euo pipefail
mkdir -p '{outdir}' '{LOGS}'
export PYTHONPATH={ROOT}/src
export HF_ENDPOINT={HF_ENDPOINT}
export HF_HOME={HF_HOME}
export CUDA_VISIBLE_DEVICES={p['gpu']}
nohup {PY} {ROOT}/src/train.py \\
  --job-id {p['job_id']} --arch-id {p['arch_id']} --c-flops {p['c_flops']} --seq-len 2048 \\
  --train-bin {train_bin} --val-bin {val_bin} \\
  --train-bytes {train_bytes} --val-bytes {val_bytes} \\
  --out-dir {outdir} \\
  --batch-tokens {p['batch_tokens']} --max-lr {p['max_lr']} --seed {p['seed']} \\
  > {log} 2>&1 &
echo $! > {outdir}/pid
"""
        ssh(p["host"], script)
        print(f"[launch] {p['job_id']} -> {p['host']} gpu{p['gpu']}")
    print(f"[launch] {len(plan)} jobs started")


if __name__ == "__main__":
    main()
