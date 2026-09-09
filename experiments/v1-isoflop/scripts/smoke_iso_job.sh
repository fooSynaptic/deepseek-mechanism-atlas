#!/usr/bin/env bash
# 8-GPU smoke for one IsoFLOP job_id. Writes results/<job_id>/smoke_ok on success.
# Does NOT write last.pt into the real job dir (uses <job_id>.smoke).
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
JOB_ID=${1:?usage: smoke_iso_job.sh <job_id> [host]}
HOST=${2:-}
STEPS=${SMOKE_STEPS:-8}
GRID=${DS_GRID:-$SCRIPT_DIR/../configs/grid_wave_l2.json}

python3 - "$JOB_ID" "$GRID" "$HOST" "$STEPS" "$DS_ROOT" "$DS_PY" "$DS_DATA" "$DS_RESULTS" "$DS_LOGS" <<'PY'
import json, os, subprocess, sys, zlib
job_id, grid_path, host, steps, root, py, data, results, logs = sys.argv[1:]
steps = int(steps)
grid = json.loads(open(grid_path).read())
job = next((j for j in grid["jobs"] if j["job_id"] == job_id), None)
if not job:
    raise SystemExit(f"job {job_id} not in {grid_path}")
if not host:
    nodes = os.environ.get("DS_NODES", "").strip()
    if not nodes:
        raise SystemExit("pass a host or set DS_NODES='host:8 ...'")
    host = nodes.split()[0].split(":")[0]
port = 29500 + zlib.crc32(f"{job_id}_smoke".encode()) % 80
smoke_dir = f"{results}/{job_id}.smoke"
ok_dir = f"{results}/{job_id}"
ok_file = f"{ok_dir}/smoke_ok"
log = f"{logs}/{job_id}.smoke.{host}.log"
ng = int(job.get("n_gpus", 8))
cvd = ",".join(str(i) for i in range(ng))
cmd = f"""
set -euo pipefail
mkdir -p '{smoke_dir}' '{ok_dir}' '{logs}'
export PYTHONPATH={root}/src
export CUDA_VISIBLE_DEVICES={cvd}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
{py} -m torch.distributed.run --nnodes=1 --nproc_per_node={ng} \\
  --master_addr=127.0.0.1 --master_port={port} --max-restarts=0 \\
  {root}/src/train.py \\
  --job-id {job_id}_smoke --arch-id {job['arch_id']} --c-flops {job['c_flops']} \\
  --seq-len 4096 --no-checkpoint --no-resume --smoke-steps {steps} \\
  --train-bin {data}/train.bin --val-bin {data}/val.bin \\
  --train-bytes {data}/train.bytes.bin --val-bytes {data}/val.bytes.bin \\
  --out-dir {smoke_dir} --batch-tokens {job['batch_tokens']} --max-lr {job['max_lr']} \\
  --seed {job.get('seed', 1)} --log-every 1 --eval-batches 1
echo SMOKE_OK > {ok_file}
echo SMOKE_OK
"""
print(f"[smoke] {job_id} host={host} steps={steps} arch={job['arch_id']}", flush=True)
r = subprocess.run(["ssh", "-o", "BatchMode=yes", host, "bash", "-s"], input=cmd, text=True)
if r.returncode != 0:
    raise SystemExit(f"smoke {job_id} failed rc={r.returncode}")
print(f"[smoke] ok {job_id} -> {ok_file}")
PY
