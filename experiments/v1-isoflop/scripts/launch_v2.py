#!/usr/bin/env python3
"""Pack IsoFLOP jobs: one independent 8-GPU single-node job per idle node.

Each job uses torchrun --nnodes=1 --nproc_per_node=N on one idle host from DS_NODES.
Up to three jobs may run concurrently (one per node). No multi-node NCCL.
Skips result.json. Checkpoints from a different world_size are not resumed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(os.environ.get("DS_ROOT", Path(__file__).resolve().parents[1]))
NODES = os.environ.get("DS_NODES", "").strip()
PY = os.environ.get(
    "DS_REMOTE_PY",
    os.environ.get("DS_PY", str(ROOT / ".venv" / "bin" / "python")),
)
DATA = Path(os.environ.get("DS_DATA", ROOT / "data"))
RESULTS = Path(os.environ.get("DS_RESULTS", ROOT / "results"))
LOGS = Path(os.environ.get("DS_LOGS", ROOT / "logs"))
BUSY_MIB = 500


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


def node_caps():
    caps = []
    for spec in NODES.split():
        host, n = spec.split(":")
        caps.append({"host": host, "gpus": int(n), "busy": set()})
    return caps


def load_busy(caps) -> None:
    for c in caps:
        out = ssh(
            c["host"],
            "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits",
        )
        for line in out.strip().splitlines():
            if not line.strip():
                continue
            idx_s, mem_s = [p.strip() for p in line.split(",")[:2]]
            idx, mem = int(idx_s), float(mem_s)
            if idx < c["gpus"] and mem > BUSY_MIB:
                c["busy"].add(idx)


def remote_flag(host: str, path: str) -> bool:
    return "YES" in ssh(host, f"test -e {path} && echo YES || echo NO")


def running_jobs(caps) -> set[str]:
    """Job ids with a live trainer, matched by --job-id rather than pid file.

    PID numbers get recycled and stale pid files survive aborted generations, so
    `kill -0 $(cat pid)` both false-positives (blocks a queue forever) and
    false-negatives (launches a duplicate onto the same GPUs).

    One ssh per host: probing per job floods sshd's MaxStartups and stalls.
    """
    found: set[str] = set()
    for c in caps:
        out = ssh(
            c["host"],
            "pgrep -af -- '--job-id' 2>/dev/null | "
            "grep -o -- '--job-id [A-Za-z0-9_]*' | awk '{print $2}' | sort -u || true",
        )
        found.update(line.strip() for line in out.splitlines() if line.strip())
    return found


def finished_jobs(head: str, job_ids: list[str]) -> set[str]:
    """One ssh for the whole grid instead of one stat per job."""
    ids = " ".join(job_ids)
    out = ssh(
        head,
        f"for j in {ids}; do [ -e {RESULTS}/$j/result.json ] && echo $j; done; true",
    )
    return {line.strip() for line in out.splitlines() if line.strip()}


def remote_world(head: str, meta_path: Path) -> int | None:
    out = ssh(
        head,
        f"python3 -c \"import json,os,sys; p='{meta_path}'; "
        f"print(json.load(open(p)).get('world_size','') if os.path.isfile(p) else '')\"",
    ).strip()
    if not out:
        return None
    try:
        return int(out)
    except ValueError:
        return None


def master_port(job_id: str) -> int:
    # zlib.crc32 is stable across processes; builtin hash() is seed-randomized,
    # so a duplicate launch of the same job would pick a different port and
    # silently succeed instead of colliding.
    return 29500 + zlib.crc32(job_id.encode()) % 80


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--max-jobs",
        type=int,
        default=3,
        help="max concurrent single-node jobs (one per node; default 3)",
    )
    p.add_argument(
        "--grid",
        type=str,
        default="",
        help="grid json (default: $DS_GRID or configs/grid.json)",
    )
    p.add_argument(
        "--require-smoke",
        action="store_true",
        default=True,
        help="only launch a job after results/<id>/smoke_ok exists",
    )
    p.add_argument("--no-require-smoke", action="store_false", dest="require_smoke")
    args = p.parse_args()
    if not NODES:
        raise SystemExit(
            "Set DS_NODES='host:ngpus host2:ngpus' for this SSH packer, "
            "or use scripts/run_wave.py for a local sequential run."
        )
    if args.max_jobs < 1:
        raise SystemExit("--max-jobs must be >= 1")

    local = Path(__file__).resolve().parents[1]
    grid_path = Path(args.grid or os.environ.get("DS_GRID", str(local / "configs" / "grid.json")))
    if not grid_path.is_absolute():
        grid_path = local / grid_path
    grid = json.loads(grid_path.read_text())
    print(f"[grid] {grid_path}")
    train_bin = str(DATA / "train.bin")
    val_bin = str(DATA / "val.bin")
    caps = node_caps()
    head = caps[0]["host"]
    chk = ssh(head, f"test -f {train_bin} && test -f {val_bin} && echo OK")
    if "OK" not in chk:
        raise SystemExit(f"missing corpus {train_bin}")

    nbytes = int(ssh(head, f"stat -c%s {train_bin}").strip())
    n_train = nbytes // 2
    jobs = list(grid["jobs"])
    max_d = max(int(j["D_tokens"]) for j in jobs)
    print(f"[data] unique_train={n_train} max_D={max_d}")

    load_busy(caps)
    print(
        "[gpus] "
        + " ".join(
            f"{c['host']}:busy={sorted(c['busy']) or '-'} cap={c['gpus']}" for c in caps
        )
    )

    done = finished_jobs(head, [j["job_id"] for j in jobs])
    live = running_jobs(caps)
    skipped = [j["job_id"] for j in jobs if j["job_id"] in done]
    inflight = [j["job_id"] for j in jobs if j["job_id"] not in done and j["job_id"] in live]

    wait_data, wait_smoke, launchable = [], [], []
    for j in jobs:
        jid = j["job_id"]
        if jid in skipped or jid in inflight:
            continue
        if int(j["D_tokens"]) > n_train:
            wait_data.append(jid)
            print(f"[wait-data] {jid} D={j['D_tokens']} > unique={n_train}")
            continue
        smoke_p = str(RESULTS / jid / "smoke_ok")
        if args.require_smoke and not remote_flag(head, smoke_p):
            wait_smoke.append(jid)
            print(f"[wait-smoke] {jid}")
            continue
        launchable.append(j)

    free_nodes = [c for c in caps if not c["busy"]]
    slots = max(0, min(len(free_nodes), args.max_jobs - len(inflight)))
    print(
        f"[pack] inflight={inflight or '-'} free_nodes={len(free_nodes)} "
        f"slots={slots} launchable={[j['job_id'] for j in launchable] or '-'} "
        f"wait_smoke={wait_smoke or '-'} wait_data={wait_data or '-'}"
    )

    if slots == 0 or not launchable:
        plan = {
            "launched": [],
            "deferred": [j["job_id"] for j in launchable] + wait_smoke + wait_data,
            "wait_smoke": wait_smoke,
            "wait_data": wait_data,
            "skipped": skipped,
            "running": inflight,
        }
        print(json.dumps(plan, indent=2))
        return

    launched = []
    for i in range(min(slots, len(launchable))):
        job = launchable[i]
        node = free_nodes[i]
        host = node["host"]
        ng = node["gpus"]
        world = ng
        outdir = RESULTS / job["job_id"]
        extra = ""
        prev_ws = remote_world(head, outdir / "meta.json")
        if prev_ws is not None and prev_ws != world and remote_flag(head, str(outdir / "last.pt")):
            extra = "--no-resume"
            print(f"[ckpt] {job['job_id']} last.pt world={prev_ws} != {world}; starting fresh")

        ssh(head, f"mkdir -p {outdir} {LOGS}")
        cvd = ",".join(str(i) for i in range(ng))
        port = master_port(job["job_id"])
        log = LOGS / f"{job['job_id']}.{host}.log"
        common = (
            f"--job-id {job['job_id']} --arch-id {job['arch_id']} "
            f"--c-flops {job['c_flops']} --seq-len 4096 --no-checkpoint \\\n"
            f"  --train-bin {train_bin} --val-bin {val_bin} \\\n"
            f"  --train-bytes {DATA}/train.bytes.bin --val-bytes {DATA}/val.bytes.bin \\\n"
            f"  --out-dir {outdir} --batch-tokens {job['batch_tokens']} "
            f"--max-lr {job['max_lr']} --seed {job.get('seed', 1)} {extra}"
        )
        runner = (
            f"{PY} -m torch.distributed.run --nnodes=1 --nproc_per_node={ng} "
            f"--master_addr=127.0.0.1 --master_port={port} "
            f"--max-restarts=0 {ROOT}/src/train.py"
        )
        # flock + pgrep re-check inside the critical section: two schedulers racing
        # on the same job must not both reach nohup.
        cmd = f"""
set -euo pipefail
mkdir -p '{outdir}' '{LOGS}'
exec 9>'{outdir}/.launch.lock'
if ! flock -n 9; then echo 'SKIP locked'; exit 0; fi
if pgrep -f -- '--job-id {job["job_id"]} ' >/dev/null 2>&1; then
  echo 'SKIP already running'
  exit 0
fi
export PYTHONPATH={ROOT}/src
export CUDA_VISIBLE_DEVICES={cvd}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
nohup {runner} \\
  {common} \\
  >> {log} 2>&1 &
echo $! > {outdir}/pid.{host}
cp {outdir}/pid.{host} {outdir}/pid
echo {host} {cvd} > {outdir}/slot
echo LAUNCHED $!
"""
        print(f"[launch] {job['job_id']} {host} GPU {cvd} world={world} port={port}")
        out = ssh(host, cmd)
        if "SKIP" in out:
            print(f"[skip] {job['job_id']} {out.strip()}")
            continue
        node["busy"].update(range(ng))
        launched.append(
            {
                "job_id": job["job_id"],
                "host": host,
                "gpus": cvd,
                "n_gpus": ng,
                "world": world,
            }
        )

    deferred = [j["job_id"] for j in launchable[len(launched) :]] + wait_smoke + wait_data
    plan = {
        "unique_train": n_train,
        "max_D": max_d,
        "launched": launched,
        "deferred": deferred,
        "wait_smoke": wait_smoke,
        "wait_data": wait_data,
        "skipped": skipped,
        "running": inflight + [x["job_id"] for x in launched],
    }
    print(json.dumps(plan, indent=2))
    ssh(head, f"cat > {LOGS}/launch_plan.json <<'PLAN'\n{json.dumps(plan, indent=2)}\nPLAN\n")


if __name__ == "__main__":
    main()
