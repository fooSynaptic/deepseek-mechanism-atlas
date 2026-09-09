#!/usr/bin/env bash
# Start wait_l3_arm_inner on the cluster head.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
HOST=${1:?usage: wait_l3_arm.sh <head-host>}
ROOT=$DS_ROOT
ssh -o BatchMode=yes "$HOST" bash -s <<EOF
set -euo pipefail
mkdir -p $DS_LOGS
chmod +x $ROOT/scripts/wait_l3_arm_inner.sh $ROOT/scripts/watch_l3_20gpu.sh $ROOT/scripts/launch_l3_20gpu.py
nohup bash $ROOT/scripts/wait_l3_arm_inner.sh > $DS_LOGS/wait_l3_arm.log 2>&1 &
echo \$! > $DS_LOGS/wait_l3_arm.pid
echo "wait_l3_arm pid=\$(cat $DS_LOGS/wait_l3_arm.pid)"
EOF
echo "waiter started on $HOST"
