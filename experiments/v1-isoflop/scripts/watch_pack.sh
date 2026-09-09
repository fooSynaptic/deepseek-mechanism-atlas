#!/usr/bin/env bash
# Pack IsoFLOP: fill idle nodes with independent 8-GPU single-node jobs.
# Resilient: transient ssh failures do not kill the loop.
# DS_GRID selects which job list to drain (Wave A grid.json vs Wave L).
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
ds_require_nodes
: "${DS_REMOTE_PY:=$DS_PY}"
export DS_REMOTE_PY
HEAD=${HEAD:-${DS_NODES%%:*}}
SLEEP=${WATCH_SLEEP:-90}
MAX_JOBS=${MAX_JOBS:-3}
GRID=${DS_GRID:-$SCRIPT_DIR/../configs/grid.json}

n_jobs=$(python3 -c "import json; print(json.load(open('$GRID'))['n_jobs'])")

count_done() {
  python3 -c "
import json, os, subprocess, sys
grid=json.load(open('$GRID'))
ids=[j['job_id'] for j in grid['jobs']]
root=os.environ.get('DS_RESULTS','')
if root and os.path.isdir(root):
    print(sum(os.path.isfile(os.path.join(root,j,'result.json')) for j in ids))
    sys.exit(0)
cmd='n=0; for j in '+' '.join(ids)+f'; do [ -e {root}/\$j/result.json ] && n=\$((n+1)); done; echo \$n'
r=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','$HEAD',cmd],capture_output=True,text=True)
print((r.stdout or '0').strip() or '0')
" 2>/dev/null || echo 0
}

while true; do
  n_done=$(count_done)
  n_done=${n_done:-0}
  echo "[watch] grid=$GRID done=$n_done/$n_jobs $(date -u +%H:%M:%SZ)"
  if [[ "$n_done" -ge "$n_jobs" ]]; then
    echo "[watch] all result.json present"
    exit 0
  fi
  if ! python3 "$SCRIPT_DIR/launch_v2.py" --max-jobs "$MAX_JOBS" --grid "$GRID"; then
    echo "[watch] launch_v2 failed; will retry"
  fi
  sleep "$SLEEP"
done
