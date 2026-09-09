#!/usr/bin/env bash
# SIGTERM train jobs. Default: one job_id. Pass --all to stop every ds-v1-isoflop train.
# Never kill -9.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <job_id> | $0 --all" >&2
  echo "one experiment at a time: stop a single job, wait, then relaunch." >&2
  exit 2
fi

if [[ "$1" != "--all" ]]; then
  exec "$SCRIPT_DIR/stop_job.sh" "$1"
fi

if [[ -z "${DS_NODES}" ]]; then
  mapfile -t lines < <(ps -eo pid,args | awk '/ds-v1-isoflop/ && /src\/train.py/ {print}')
  if [[ ${#lines[@]} -eq 0 ]]; then
    echo "(no local train processes)"
    exit 0
  fi
  for line in "${lines[@]}"; do
    pid=$(echo "$line" | awk '{print $1}')
    echo "TERM pid=$pid"
    kill -TERM "$pid" || true
  done
  echo "sent SIGTERM. wait, then check GPU memory."
  exit 0
fi

for spec in $DS_NODES; do
  host=${spec%%:*}
  echo "=== stop $host ==="
  mapfile -t lines < <(ssh -o BatchMode=yes "$host" 'ps -eo pid,args' | awk '/ds-v1-isoflop/ && /src\/train.py/ {print}')
  if [[ ${#lines[@]} -eq 0 ]]; then
    echo "(no train processes)"
    continue
  fi
  for line in "${lines[@]}"; do
    pid=$(echo "$line" | awk '{print $1}')
    echo "TERM $host pid=$pid"
    ssh -o BatchMode=yes "$host" "kill -TERM $pid" || true
  done
done
echo "sent SIGTERM. wait, then preflight GPUs."
