#!/usr/bin/env bash
# Copy repo to a remote shared filesystem via tar over SSH.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
HOST=${1:?usage: sync_to_fs.sh <host>}
ssh -o BatchMode=yes "$HOST" "mkdir -p $DS_ROOT"
cd "$SCRIPT_DIR/.."
tar czf - \
  --exclude '.venv' --exclude 'data' --exclude 'results' --exclude 'logs' \
  --exclude '.hf' --exclude '__pycache__' --exclude '.git' \
  . | ssh -o BatchMode=yes "$HOST" "tar xzf - -C $DS_ROOT"
echo "synced -> $HOST:$DS_ROOT"
ssh -o BatchMode=yes "$HOST" "ls $DS_ROOT/src $DS_ROOT/scripts $DS_ROOT/configs | head"
