#!/usr/bin/env bash
# Drive L4 sequential jobs via launch_l4_24gpu.py. Does not start the 8-GPU watch_daemon.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
ROOT=$DS_ROOT
DATA=${DS_DATA:-$ROOT/data}
PY=${DS_PY:-$ROOT/.venv/bin/python}
GRID=${DS_GRID:-$ROOT/configs/grid_wave_l4.json}
export DS_ROOT=$ROOT DS_DATA=$DATA DS_PY=$PY DS_GRID=$GRID
export PYTHONUNBUFFERED=1
cd "$ROOT"
mkdir -p "$ROOT/logs"

echo "[watch-l4-24] start $(date -u +%Y-%m-%dT%H:%M:%SZ)"

JOBS=(
  iso_c3e18_u0 iso_c3e18_u1 iso_c3e18_u2 iso_c3e18_u3
  iso_c1e19_u1 iso_c1e19_u2 iso_c1e19_u3
  iso_c3e19_u3
)

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

run_one() {
  local jid=$1
  if [[ -f $ROOT/results/$jid/result.json ]]; then
    echo "[watch-l4-24] skip done $jid"
    return 0
  fi
  if job_alive "$jid"; then
    echo "[watch-l4-24] attach existing full $jid"
    $PY -u $ROOT/scripts/launch_l4_24gpu.py --job-id "$jid" --grid "$GRID" --wait
    return 0
  fi
  echo "[watch-l4-24] smoke $jid"
  $PY -u $ROOT/scripts/launch_l4_24gpu.py --job-id "$jid" --grid "$GRID" --smoke
  sleep 10
  echo "[watch-l4-24] full $jid"
  $PY -u $ROOT/scripts/launch_l4_24gpu.py --job-id "$jid" --grid "$GRID" --wait
}

for jid in "${JOBS[@]}"; do
  run_one "$jid"
done
echo "[watch-l4-24] all done $(date -u +%Y-%m-%dT%H:%M:%SZ)"
