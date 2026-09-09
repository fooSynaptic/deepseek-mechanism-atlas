#!/usr/bin/env bash
# Shared paths. Override before sourcing.
#   export DS_ROOT=/path/to/experiments/v1-isoflop
#   export DS_NODES='node-a:8 node-b:8 node-c:8'

_ds_lib=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
_ds_scripts=$(cd "$_ds_lib/.." && pwd)
_ds_repo=$(cd "$_ds_scripts/.." && pwd)

: "${DS_ROOT:=$_ds_repo}"
: "${DS_NODES:=}"
: "${DS_VENV:=$DS_ROOT/.venv}"
: "${DS_PY:=$DS_VENV/bin/python}"
if [[ ! -x "$DS_PY" ]]; then
  DS_PY=python3
fi
: "${DS_DATA:=$DS_ROOT/data}"
: "${DS_RESULTS:=$DS_ROOT/results}"
: "${DS_LOGS:=$DS_ROOT/logs}"
: "${HF_HOME:=$DS_ROOT/.hf}"
: "${HF_DATASETS_CACHE:=$DS_ROOT/.hf/datasets}"

export DS_ROOT DS_NODES DS_VENV DS_PY DS_DATA DS_RESULTS DS_LOGS
export HF_HOME HF_DATASETS_CACHE
export PYTHONPATH="$DS_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

ds_require_nodes() {
  if [[ -z "${DS_NODES:-}" ]]; then
    echo "Set DS_NODES='host-a:8 host-b:8' (SSH alias:GPU count per node)." >&2
    exit 1
  fi
}

ds_node_hosts() {
  local spec
  for spec in ${DS_NODES:-}; do
    echo "${spec%%:*}"
  done
}
