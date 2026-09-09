#!/usr/bin/env bash
# Download FineWeb-Edu sample/100BT shards whose names are not in the 10BT 14-file set.
# Writes to $DS_ROOT/data_40b/raw/fineweb_100bt_parquet. Does not touch data/train.bin.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
HOST=${1:?usage: fetch_fineweb_100bt.sh <host>}
OUT=$DS_ROOT/data_40b/raw/fineweb_100bt_parquet
ssh -o BatchMode=yes "$HOST" bash -s <<EOF
set -euo pipefail
export PYTHONUNBUFFERED=1
mkdir -p $OUT $DS_LOGS
PY=$DS_VENV/bin/python
if [[ ! -x "\$PY" ]]; then PY=python3; fi
nohup \$PY -u $DS_ROOT/scripts/fetch_fineweb_100bt.py --out-dir $OUT \\
  > $DS_LOGS/fetch_fineweb_100bt.log 2>&1 &
echo \$! > $DS_LOGS/fetch_fineweb_100bt.pid
echo "pid=\$(cat $DS_LOGS/fetch_fineweb_100bt.pid) log=$DS_LOGS/fetch_fineweb_100bt.log"
EOF
echo "100BT fetch started on $HOST"
