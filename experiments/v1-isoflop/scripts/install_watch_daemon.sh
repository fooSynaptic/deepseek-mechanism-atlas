#!/usr/bin/env bash
# Install/restart the IsoFLOP watch daemon on the cluster head (survives laptop session end).
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
ds_require_nodes
HOST=${1:?usage: install_watch_daemon.sh <head-host>}
if [[ -z "${DS_GRID:-}" ]]; then
  GRID_ABS=$DS_ROOT/configs/grid.json
elif [[ "$DS_GRID" == /* ]]; then
  GRID_ABS=$DS_GRID
else
  GRID_ABS=$DS_ROOT/$DS_GRID
fi

# Sync scripts first so remote has the latest daemon.
bash "$SCRIPT_DIR/sync_to_fs.sh" "$HOST"

ssh -o BatchMode=yes "$HOST" "bash -s" <<EOF
set -euo pipefail
ROOT=$DS_ROOT
export DS_GRID=$GRID_ABS
export DS_DATA=${DS_DATA}
cd "\$ROOT"
# Stop stale schedulers. Killing watch_daemon orphans its watch_pack child, and a
# surviving orphan plus a fresh daemon means two schedulers launching the same
# job onto the same GPUs -- so keep signalling until none are left.
for attempt in 1 2 3 4 5 6 7 8 9 10; do
  pids=\$(pgrep -f 'watch[_]pack.sh|watch[_]daemon.sh|launch[_]v2.py|run[_]watch_remote.sh' | grep -v "^\$\$\$\$\$" || true)
  [[ -z "\$pids" ]] && break
  echo "[install] stopping stale schedulers: \$(echo \$pids | tr '\n' ' ')"
  for p in \$pids; do kill -TERM "\$p" 2>/dev/null || true; done
  sleep 3
done
leftover=\$(pgrep -f 'watch[_]pack.sh|watch[_]daemon.sh|launch[_]v2.py' || true)
if [[ -n "\$leftover" ]]; then
  echo "FAIL stale schedulers survived: \$leftover" >&2
  exit 1
fi
rm -f \$ROOT/logs/watch_daemon.pid
chmod +x \$ROOT/scripts/watch_daemon.sh \$ROOT/scripts/watch_pack.sh
# Fully detach: new session, ignore hangup, close stdio
nohup setsid bash \$ROOT/scripts/watch_daemon.sh </dev/null >>\$ROOT/logs/watch_daemon.log 2>&1 &
sleep 2
pid=\$(cat \$ROOT/logs/watch_daemon.pid 2>/dev/null || true)
if [[ -n "\$pid" ]] && kill -0 "\$pid" 2>/dev/null; then
  echo "OK daemon pid=\$pid host=\$(hostname)"
  pgrep -fa 'watch_daemon|watch_pack' | head -5
  tail -5 \$ROOT/logs/watch_daemon.log
else
  echo "FAIL daemon did not start" >&2
  tail -20 \$ROOT/logs/watch_daemon.log >&2 || true
  exit 1
fi
EOF
