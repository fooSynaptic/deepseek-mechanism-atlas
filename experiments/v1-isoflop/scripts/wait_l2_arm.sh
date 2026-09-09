#!/usr/bin/env bash
# On the head node: wait for data_l2/meta.json, then restart the IsoFLOP
# daemon with DS_DATA=data_l2 so jobs whose D fits unique tokens can launch.
# Does not stop fetch_fineweb_100bt.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
HOST=${1:?usage: wait_l2_arm.sh <head-host>}
ROOT=$DS_ROOT

ssh -o BatchMode=yes "$HOST" bash -s <<EOF
set -euo pipefail
mkdir -p $DS_LOGS
nohup bash $ROOT/scripts/wait_l2_arm_inner.sh \\
  > $DS_LOGS/wait_l2_arm.log 2>&1 &
echo \$! > $DS_LOGS/wait_l2_arm.pid
echo "wait_l2_arm pid=\$(cat $DS_LOGS/wait_l2_arm.pid)"
EOF
echo "waiter started on $HOST"
