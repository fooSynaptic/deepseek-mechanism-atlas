#!/usr/bin/env bash
# Runs on the cluster head. Poll data_l2/meta.json then restart watch_daemon with DS_DATA=data_l2.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
ds_require_nodes
ROOT=$DS_ROOT
META=$ROOT/data_l2/meta.json
echo "[wait-l2] polling $META"
while [[ ! -f $META ]]; do sleep 30; done
n=$(python3 -c "import json; print(json.load(open('$META'))['n_train'])")
echo "[wait-l2] n_train=$n $(date -u +%Y-%m-%dT%H:%M:%SZ)"
export DS_ROOT=$ROOT
export DS_DATA=$ROOT/data_l2
export DS_GRID=$ROOT/configs/grid_wave_l2.json
export DS_PY=$ROOT/.venv/bin/python
export DS_REMOTE_PY=$ROOT/.venv/bin/python
export DS_RESULTS=$ROOT/results
export DS_LOGS=$ROOT/logs
export MAX_JOBS=3
export WATCH_SLEEP=90
cd "$ROOT"
self=$$
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  pids=$(pgrep -f 'watch[_]pack.sh|watch[_]daemon.sh|launch[_]v2.py|run[_]watch_remote.sh' || true)
  pids=$(echo "$pids" | grep -v "^${self}$" || true)
  [[ -z "$pids" ]] && break
  echo "[wait-l2] stopping stale schedulers: $pids"
  for p in $pids; do kill -TERM "$p" 2>/dev/null || true; done
  sleep 3
done
rm -f "$ROOT/logs/watch_daemon.pid"
nohup setsid bash "$ROOT/scripts/watch_daemon.sh" </dev/null >>"$ROOT/logs/watch_daemon.log" 2>&1 &
sleep 2
echo "[wait-l2] daemon pid=$(cat $ROOT/logs/watch_daemon.pid) DS_DATA=$DS_DATA n_train=$n"
