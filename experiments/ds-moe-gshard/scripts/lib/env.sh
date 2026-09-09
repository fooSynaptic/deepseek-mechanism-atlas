#!/usr/bin/env bash
# Shared paths for this experiment directory.
_ds_lib=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_ds_scripts=$(cd "$_ds_lib/.." && pwd)
_ds_repo=$(cd "$_ds_scripts/.." && pwd)

: "${DS_ROOT:=$_ds_repo}"
: "${DS_RESULTS:=$DS_ROOT/results}"
: "${DS_LOGS:=$DS_ROOT/logs}"
: "${DS_PY:=python3}"

export DS_ROOT DS_RESULTS DS_LOGS DS_PY
export PYTHONPATH="$DS_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Optional private overrides (gitignored). Do not commit host maps or addresses.
if [[ -f "$_ds_lib/env.local.sh" ]]; then
  # shellcheck source=/dev/null
  source "$_ds_lib/env.local.sh"
fi
