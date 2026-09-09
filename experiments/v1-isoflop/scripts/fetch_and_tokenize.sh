#!/usr/bin/env bash
# Fetch open mix then tokenize locally. Prefer scripts/prepare_data.sh.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$SCRIPT_DIR/prepare_data.sh" "${1:-wave-a}"
