#!/usr/bin/env bash
# Local status (no SSH). For a remote packer, set DS_NODES and use scripts/status.py on the head.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
exec "$DS_PY" "$SCRIPT_DIR/status.py" "$@"
