#!/usr/bin/env bash
# Keep watch_pack.sh alive on the cluster head until all IsoFLOP jobs finish.
# Auto-restarts on crash; single-instance via pidfile.
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
ds_require_nodes
ROOT=$DS_ROOT
LOG_DIR=${DS_LOGS:-$ROOT/logs}
PIDFILE=$LOG_DIR/watch_daemon.pid
LOG=$LOG_DIR/watch_daemon.log
PACK=${WATCH_PACK:-$SCRIPT_DIR/watch_pack.sh}

mkdir -p "$LOG_DIR"

if [[ -f "$PIDFILE" ]]; then
  old=$(cat "$PIDFILE" 2>/dev/null || true)
  if [[ -n "${old:-}" ]] && kill -0 "$old" 2>/dev/null; then
    # Already a live daemon; if argv matches, stay out.
    if tr '\0' ' ' < /proc/"$old"/cmdline 2>/dev/null | grep -q watch_daemon; then
      echo "[daemon] already running pid=$old"
      exit 0
    fi
  fi
fi

echo $$ > "$PIDFILE"
# Take the child down with us: an orphaned watch_pack keeps scheduling and can
# double-launch a job alongside a freshly installed daemon.
child=""
cleanup() {
  [[ -n "$child" ]] && kill -TERM "$child" 2>/dev/null
  rm -f "$PIDFILE"
}
trap cleanup EXIT
trap 'cleanup; exit 143' TERM INT

export DS_ROOT=$ROOT
export DS_REMOTE_PY=${DS_REMOTE_PY:-$ROOT/.venv/bin/python}
export DS_PY=${DS_PY:-$ROOT/.venv/bin/python}
export DS_DATA=${DS_DATA:-$ROOT/data}
export DS_RESULTS=${DS_RESULTS:-$ROOT/results}
export DS_LOGS=$LOG_DIR
export WATCH_SLEEP=${WATCH_SLEEP:-90}
export MAX_JOBS=${MAX_JOBS:-3}
export DS_GRID=${DS_GRID:-$ROOT/configs/grid.json}
export PYTHONPATH=$ROOT/src${PYTHONPATH:+:$PYTHONPATH}
cd "$ROOT"

n_jobs=$(python3 -c "import json; print(json.load(open('$DS_GRID'))['n_jobs'])")

echo "[daemon] start pid=$$ $(date -u +%Y-%m-%dT%H:%M:%SZ) n_jobs=$n_jobs" | tee -a "$LOG"

while true; do
  n_done=$(python3 -c "import json,os; g=json.load(open('$DS_GRID')); r='$DS_RESULTS'; print(sum(os.path.isfile(os.path.join(r,j['job_id'],'result.json')) for j in g['jobs']))")
  n_done=${n_done:-0}
  if [[ "$n_done" -ge "$n_jobs" ]]; then
    echo "[daemon] all done ($n_done/$n_jobs); exiting" | tee -a "$LOG"
    exit 0
  fi
  echo "[daemon] spawn watch_pack $(date -u +%H:%M:%SZ) done=$n_done/$n_jobs" | tee -a "$LOG"
  bash "$PACK" >>"$LOG" 2>&1 &
  child=$!
  wait "$child"
  rc=$?
  child=""
  echo "[daemon] watch_pack exited rc=$rc $(date -u +%H:%M:%SZ)" | tee -a "$LOG"
  if [[ "$rc" -eq 0 ]]; then
    # clean finish (all jobs done)
    exit 0
  fi
  sleep 10
done
