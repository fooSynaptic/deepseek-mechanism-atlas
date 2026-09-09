#!/usr/bin/env bash
# SIGTERM one IsoFLOP job by id on the node(s) that own it. Never kill -9.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <job_id>" >&2
  exit 2
fi
jid=$1
sent=0
if [[ -z "${DS_NODES}" ]]; then
  pid_f=$DS_RESULTS/$jid/pid
  if [[ -f "$pid_f" ]]; then
    pid=$(cat "$pid_f")
    if kill -0 "$pid" 2>/dev/null; then
      echo "TERM pid=$pid file=$pid_f"
      kill -TERM "$pid" || true
      sent=1
    fi
  fi
  if [[ $sent -eq 0 ]]; then
    echo "no live local pid for $jid"
    exit 1
  fi
  echo "sent SIGTERM to $jid. wait for last.pt / GPU memory drop. do not kill -9."
  exit 0
fi
for spec in $DS_NODES; do
  host=${spec%%:*}
  alive=$(ssh -o BatchMode=yes "$host" "bash -s" <<EOF
set -euo pipefail
sent=NO
term_if_job() {
  local f=\$1
  [[ -f "\$f" ]] || return 0
  local pid
  pid=\$(cat "\$f")
  if kill -0 "\$pid" 2>/dev/null; then
    if tr '\\0' ' ' < /proc/\$pid/cmdline | grep -q -- "$jid"; then
      echo "TERM pid=\$pid file=\$f"
      kill -TERM "\$pid" || true
      sent=YES
    fi
  fi
}
term_if_job $DS_RESULTS/$jid/pid.$host
term_if_job $DS_RESULTS/$jid/pid
echo \$sent
EOF
)
  if grep -q YES <<<"$alive"; then
    echo "[$host] $alive"
    sent=1
  fi
done

if [[ $sent -eq 0 ]]; then
  echo "no live pid for $jid on whitelist nodes"
  exit 1
fi
echo "sent SIGTERM to $jid. wait for last.pt / GPU memory drop. do not kill -9."
