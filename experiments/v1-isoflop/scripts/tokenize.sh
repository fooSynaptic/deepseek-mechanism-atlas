#!/usr/bin/env bash
# Tokenize into $DS_DATA. Local, no SSH.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
SOURCE=${SOURCE:-jsonl-dir}
TRAIN_TOKENS=${TRAIN_TOKENS:-4000000000}
VAL_TOKENS=${VAL_TOKENS:-50000000}
JSONL_DIR=${JSONL_DIR:-$DS_DATA/raw/mix}

mkdir -p "$DS_DATA" "$HF_HOME" "$HF_DATASETS_CACHE"
export PYTHONUNBUFFERED=1
extra=()
if [[ "$SOURCE" == jsonl-dir ]]; then
  extra+=(--jsonl-dir "$JSONL_DIR")
elif [[ "$SOURCE" == parquet-dir ]]; then
  extra+=(--parquet-dir "${PARQUET_DIR:?set PARQUET_DIR}")
elif [[ "$SOURCE" == local-txt ]]; then
  extra+=(--local-txt "${LOCAL_TXT:?set LOCAL_TXT}")
fi
"$DS_PY" "$SCRIPT_DIR/tokenize_corpus.py" \
  --out-dir "$DS_DATA" \
  --source "$SOURCE" \
  --train-tokens "$TRAIN_TOKENS" \
  --val-tokens "$VAL_TOKENS" \
  "${extra[@]}"
