#!/usr/bin/env bash
# Fetch a public mix and tokenize to packed GPT-2 bins under $DS_DATA.
# Presets: smoke | demo | wave-a | full
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"

PRESET=${1:-demo}
mkdir -p "$DS_DATA/raw" "$DS_LOGS" "$HF_HOME" "$HF_DATASETS_CACHE"
export PYTHONUNBUFFERED=1

case "$PRESET" in
  smoke)
    "$DS_PY" "$SCRIPT_DIR/make_smoke_data.py" --out-dir "$DS_DATA/smoke"
    echo "smoke bins -> $DS_DATA/smoke"
    exit 0
    ;;
  demo)
    TRAIN_TOKENS=${TRAIN_TOKENS:-2000000}
    VAL_TOKENS=${VAL_TOKENS:-200000}
    FETCH_PRESET=demo
    ;;
  wave-a)
    TRAIN_TOKENS=${TRAIN_TOKENS:-4000000000}
    VAL_TOKENS=${VAL_TOKENS:-50000000}
    FETCH_PRESET=wave-a
    ;;
  full)
    TRAIN_TOKENS=${TRAIN_TOKENS:-40000000000}
    VAL_TOKENS=${VAL_TOKENS:-50000000}
    FETCH_PRESET=full
    ;;
  *)
    echo "usage: $0 {smoke|demo|wave-a|full}" >&2
    exit 2
    ;;
esac

"$DS_PY" -u "$SCRIPT_DIR/fetch_open_mix.py" --out-dir "$DS_DATA/raw" --preset "$FETCH_PRESET"
"$DS_PY" -u "$SCRIPT_DIR/tokenize_corpus.py" \
  --out-dir "$DS_DATA" \
  --source jsonl-dir --jsonl-dir "$DS_DATA/raw/mix" \
  --train-tokens "$TRAIN_TOKENS" \
  --val-tokens "$VAL_TOKENS"
echo "packed bins -> $DS_DATA (train.bin / val.bin)"
