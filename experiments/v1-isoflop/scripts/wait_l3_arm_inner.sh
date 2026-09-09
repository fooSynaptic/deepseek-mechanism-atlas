#!/usr/bin/env bash
# Poll data snapshot, then run the 20-GPU L3 watcher (not the 8-GPU watch_daemon).
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
ROOT=$DS_ROOT
echo "[wait-l3] handing off to watch_l3_20gpu $(date -u +%Y-%m-%dT%H:%M:%SZ)"
# Refuse if 8-GPU scheduler is alive.
if pgrep -f 'watch[_]daemon.sh|watch[_]pack.sh' >/dev/null 2>&1; then
  echo "[wait-l3] refusing: 8-GPU watch_daemon still running" >&2
  exit 1
fi
exec bash "$ROOT/scripts/watch_l3_20gpu.sh"
