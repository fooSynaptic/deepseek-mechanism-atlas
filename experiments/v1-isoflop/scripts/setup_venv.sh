#!/usr/bin/env bash
# Local venv with PyTorch + tokenize/fetch deps. No SSH.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"

mkdir -p "$DS_ROOT"
cd "$DS_ROOT"
if [[ ! -x "$DS_VENV/bin/python" ]]; then
  python3 -m venv "$DS_VENV"
fi
"$DS_VENV/bin/pip" install -U pip
"$DS_VENV/bin/pip" install -r "$DS_ROOT/requirements.txt"
"$DS_VENV/bin/python" -c "import torch,tiktoken; print('torch', torch.__version__, 'tiktoken ok')"
echo "venv -> $DS_VENV"
