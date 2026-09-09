#!/usr/bin/env bash
# Drive L3 on the DS_RDZV_NODES set. Sequential: t1 then t0.
# Does not start the 8-GPU watch_daemon.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
ROOT=$DS_ROOT
DATA=${DS_DATA:-$ROOT/data}
PY=${DS_PY:-$ROOT/.venv/bin/python}
LOG=$ROOT/logs/watch_l3_20gpu.log
NEED_T1=18410701089
NEED_T0=31165827334
export DS_ROOT=$ROOT DS_DATA=$DATA DS_PY=$PY
export PYTHONUNBUFFERED=1
cd "$ROOT"
mkdir -p "$ROOT/logs"

job_alive() {
  local jid=$1 host
  if pgrep -af train.py | grep -- "--job-id ${jid} " | grep -v grep >/dev/null 2>&1; then
    return 0
  fi
  for host in $(ds_node_hosts); do
    ssh -o BatchMode=yes "$host" "pgrep -af train.py | grep -- '--job-id ${jid} ' | grep -v grep" >/dev/null 2>&1 && return 0
  done
  return 1
}

echo "[watch-l3-20] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"

wait_unique() {
  local need=$1
  while true; do
    n=$(( $(stat -c%s "$DATA/train.bin") / 2 ))
    echo "[watch-l3-20] unique=$n need=$need $(date -u +%H:%M:%SZ)"
    [[ "$n" -ge "$need" ]] && return 0
    sleep 30
  done
}

wait_bytes() {
  while true; do
    nt=$(( $(stat -c%s "$DATA/train.bin") / 2 ))
    if [[ -f $DATA/train.bytes.bin ]]; then
      nb=$(stat -c%s "$DATA/train.bytes.bin")
    else
      nb=0
    fi
    echo "[watch-l3-20] bytes=$nb want=$nt $(date -u +%H:%M:%SZ)"
    [[ "$nb" -eq "$nt" ]] && return 0
    sleep 20
  done
}

run_one() {
  local jid=$1
  if [[ -f $ROOT/results/$jid/result.json ]]; then
    echo "[watch-l3-20] skip done $jid"
    return 0
  fi
  # If full is already live (watcher restart mid-job), skip smoke and attach --wait.
  if job_alive "$jid"; then
    echo "[watch-l3-20] attach existing full $jid"
    $PY -u $ROOT/scripts/launch_l3_20gpu.py --job-id "$jid" --wait
    return 0
  fi
  echo "[watch-l3-20] smoke $jid"
  $PY -u $ROOT/scripts/launch_l3_20gpu.py --job-id "$jid" --smoke
  sleep 10
  echo "[watch-l3-20] full $jid"
  $PY -u $ROOT/scripts/launch_l3_20gpu.py --job-id "$jid" --wait
}

wait_unique "$NEED_T1"
wait_bytes
run_one iso_c3e19_t1
# Data gate for t0 (already satisfied today; still re-check + bytes).
wait_unique "$NEED_T0"
wait_bytes
run_one iso_c3e19_t0
echo "[watch-l3-20] all done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
