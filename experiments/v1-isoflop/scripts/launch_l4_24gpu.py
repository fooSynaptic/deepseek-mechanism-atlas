#!/usr/bin/env python3
"""Launch one L4 IsoFLOP job on 24 GPUs (8+8+8) via c10d + --local-addr.

Forces --micro-seqs 1 so Formula 1 batch stays near target when world=24.
Set DS_RDZV_NODES and L4_MASTER_ADDR (defaults to the first node's IP).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import zlib
from pathlib import Path

ROOT = Path(os.environ.get("DS_ROOT", Path(__file__).resolve().parents[1]))
PY = os.environ.get("DS_PY", str(ROOT / ".venv" / "bin" / "python"))
DATA = Path(os.environ.get("DS_DATA", ROOT / "data"))
RESULTS = Path(os.environ.get("DS_RESULTS", ROOT / "results"))
LOGS = Path(os.environ.get("DS_LOGS", ROOT / "logs"))
NCCL_IFACE = os.environ.get("NCCL_SOCKET_IFNAME", "eth0")
GLOO_IFACE = os.environ.get("GLOO_SOCKET_IFNAME", "eth0")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.rdzv_nodes import load_rdzv_nodes, master_addr  # noqa: E402

NODES = load_rdzv_nodes()
MASTER = master_addr(NODES, "L4_MASTER_ADDR")


def ssh(host: str, script: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, "bash", "-s"],
        input=script,
        text=True,
        capture_output=True,
    )
    if check and r.returncode != 0:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"ssh {host} failed rc={r.returncode}")
    return r


def master_port(job_id: str, *, smoke: bool) -> int:
    key = f"{job_id}:{'s24' if smoke else 'full24'}:v1"
    return 29700 + zlib.crc32(key.encode()) % 80


def load_job(grid_path: Path, job_id: str) -> dict:
    grid = json.loads(grid_path.read_text())
    job = next((j for j in grid["jobs"] if j["job_id"] == job_id), None)
    if not job:
        raise SystemExit(f"{job_id} not in {grid_path}")
    return job


def already_running(job_id: str) -> bool:
    for host, _, _, _ in NODES:
        r = ssh(
            host,
            f"pgrep -af train.py | grep -- '--job-id {job_id} ' | grep -v grep || true",
            check=False,
        )
        if r.stdout.strip():
            return True
    return False


def wait_job_gone(job_id: str, timeout_s: int = 180) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        if not already_running(job_id):
            print(f"[l4-24] job gone {job_id}", flush=True)
            return
        time.sleep(3)
    raise SystemExit(f"timeout waiting for {job_id} processes to exit")


def wait_l4_gpus_idle(timeout_s: int = 300) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        busy = []
        for host, _, _, _ in NODES:
            r = ssh(
                host,
                "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits",
                check=False,
            )
            for line in r.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 2:
                    continue
                try:
                    used = float(parts[1])
                except ValueError:
                    continue
                if used > 1500:
                    busy.append(f"{host}:gpu{parts[0]}={used:.0f}MiB")
        if not busy:
            print("[l4-24] 24 GPUs idle", flush=True)
            return
        print(f"[l4-24] waiting GPU drain: {', '.join(busy[:4])}", flush=True)
        time.sleep(5)
    raise SystemExit("timeout waiting for L4 GPUs to drain")


def launch(job: dict, *, smoke: bool) -> None:
    jid = job["job_id"]
    port = master_port(jid, smoke=smoke)
    rdzv = f"{MASTER}:{port}"
    rdzv_id = f"{jid}_{'s24' if smoke else 'full24'}_{int(time.time())}"
    outdir = RESULTS / (f"{jid}.smoke24" if smoke else jid)
    extra = (
        "--no-resume --smoke-steps 8 --eval-batches 1 --log-every 1 --micro-seqs 1"
        if smoke
        else "--no-resume --micro-seqs 1"
    )
    LOGS.mkdir(parents=True, exist_ok=True)
    ssh(NODES[0][0], f"mkdir -p '{outdir}' '{LOGS}'")
    if not smoke:
        # Drop any 8-GPU abort leftovers; world_size=24 must start fresh.
        ssh(
            NODES[0][0],
            f"rm -f '{outdir}/last.pt' '{outdir}/result.json' '{outdir}/train.jsonl'",
        )

    common = (
        f"--job-id {jid} --arch-id {job['arch_id']} "
        f"--c-flops {job['c_flops']} --seq-len 4096 --no-checkpoint "
        f"--train-bin {DATA}/train.bin --val-bin {DATA}/val.bin "
        f"--train-bytes {DATA}/train.bytes.bin --val-bytes {DATA}/val.bytes.bin "
        f"--out-dir {outdir} --batch-tokens {job['batch_tokens']} "
        f"--max-lr {job['max_lr']} --seed {job.get('seed', 1)} {extra}"
    )

    print(f"[l4-24] {jid} smoke={smoke} rdzv={rdzv} out={outdir}", flush=True)
    for host, _, _, _ in NODES:
        log = LOGS / f"{jid}.{host}.24gpu{'.smoke' if smoke else ''}.log"
        ssh(NODES[0][0], f"mkdir -p '{LOGS}'; : > '{log}'")
    for host, nproc, cvd, local_addr in NODES:
        log = LOGS / f"{jid}.{host}.24gpu{'.smoke' if smoke else ''}.log"
        cmd = f"""
set -euo pipefail
mkdir -p '{outdir}' '{LOGS}'
export PYTHONPATH={ROOT}/src
export CUDA_VISIBLE_DEVICES={cvd}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME={NCCL_IFACE}
export GLOO_SOCKET_IFNAME={GLOO_IFACE}
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_NET=Socket
export NCCL_RAS_ENABLE=0
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_DEBUG=WARN
nohup {PY} -m torch.distributed.run \\
  --nnodes=3 --nproc_per_node={nproc} \\
  --rdzv-backend=c10d --rdzv-endpoint={rdzv} --rdzv-id={rdzv_id} \\
  --local-addr={local_addr} \\
  --rdzv-conf=join_timeout=600 \\
  --max-restarts=0 \\
  {ROOT}/src/train.py {common} \\
  >> {log} 2>&1 &
echo $! > {outdir}/pid.{host}
echo {host} {cvd} nproc={nproc} >> {outdir}/slot
echo LAUNCHED {host} pid=$!
"""
        out = ssh(host, cmd)
        print(out.stdout.strip(), flush=True)
        if host == NODES[0][0]:
            time.sleep(5)

    head = NODES[0][0]
    ssh(head, f"cp {outdir}/pid.{head} {outdir}/pid")


def wait_smoke(job_id: str, timeout_s: int = 600) -> None:
    log = LOGS / f"{job_id}.{NODES[0][0]}.24gpu.smoke.log"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = ssh(
            NODES[0][0],
            f"grep -E 'SMOKE_OK|DONE |Traceback|NCCL ERROR|RendezvousTimeout' {log} 2>/dev/null | tail -5 || true",
            check=False,
        )
        text = r.stdout
        if "RendezvousTimeout" in text or "NCCL ERROR" in text:
            raise SystemExit(f"smoke failed:\n{text}")
        if "Traceback" in text and "SignalException" not in text:
            raise SystemExit(f"smoke failed:\n{text}")
        alive = any(
            ssh(
                h,
                f"pgrep -af train.py | grep -- '--job-id {job_id} ' | grep -v grep >/dev/null && echo Y || echo N",
                check=False,
            ).stdout.strip()
            == "Y"
            for h, _, _, _ in NODES
        )
        ok = ssh(
            NODES[0][0],
            f"test -f {RESULTS}/{job_id}.smoke24/result.json && echo Y || echo N",
            check=False,
        ).stdout.strip()
        if ok == "Y":
            ssh(
                NODES[0][0],
                f"mkdir -p {RESULTS}/{job_id}; echo SMOKE24_OK > {RESULTS}/{job_id}/smoke24_ok; echo SMOKE24_OK > {RESULTS}/{job_id}/smoke_ok",
            )
            print(f"[l4-24] smoke ok {job_id}", flush=True)
            wait_job_gone(job_id, timeout_s=180)
            wait_l4_gpus_idle(timeout_s=300)
            return
        if not alive and ok != "Y":
            tail = ssh(NODES[0][0], f"tail -40 {log}", check=False).stdout
            raise SystemExit(f"smoke processes exited without result.json\n{tail}")
        time.sleep(5)
    raise SystemExit("smoke timeout")


def wait_result(job_id: str) -> None:
    p = RESULTS / job_id / "result.json"
    print(f"[l4-24] waiting {p}", flush=True)
    dead_ticks = 0
    while True:
        r = ssh(NODES[0][0], f"test -f {p} && echo Y || echo N", check=False)
        if r.stdout.strip() == "Y":
            print(f"[l4-24] done {job_id}", flush=True)
            wait_job_gone(job_id, timeout_s=180)
            wait_l4_gpus_idle(timeout_s=300)
            return
        if already_running(job_id):
            dead_ticks = 0
        else:
            dead_ticks += 1
            if dead_ticks >= 3:
                raise SystemExit(
                    f"{job_id} processes gone but {p} missing — full run died; not advancing pipe"
                )
        time.sleep(30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", required=True)
    ap.add_argument("--grid", default=str(ROOT / "configs" / "grid_wave_l4.json"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--wait", action="store_true", help="wait until result.json")
    args = ap.parse_args()
    grid_path = Path(args.grid)
    if not grid_path.is_absolute():
        grid_path = ROOT / grid_path
    job = load_job(grid_path, args.job_id)
    if already_running(args.job_id):
        print(f"[l4-24] already running {args.job_id}", flush=True)
        if args.smoke:
            print("[l4-24] skip smoke; job already live", flush=True)
            return
        if args.wait:
            wait_result(args.job_id)
        return
    wait_l4_gpus_idle(timeout_s=300)
    launch(job, smoke=args.smoke)
    if args.smoke:
        wait_smoke(args.job_id)
    elif args.wait:
        wait_result(args.job_id)


if __name__ == "__main__":
    main()
