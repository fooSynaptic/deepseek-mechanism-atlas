#!/usr/bin/env bash
# L2: grow unique train to ≥12B into $DS_ROOT/data_l2 (does not overwrite Wave A 4B bins).
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
HOST=${1:?usage: fetch_and_tokenize_l2.sh <host>}
ssh -o BatchMode=yes "$HOST" bash -s <<EOF
set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_ENDPOINT=\${HF_ENDPOINT:-https://huggingface.co}
export HF_HOME=$DS_ROOT/.hf
export HF_DATASETS_CACHE=$DS_ROOT/.hf/datasets
PY=$DS_VENV/bin/python
OUT=$DS_ROOT/data_l2
mkdir -p \$OUT/raw $DS_LOGS \$HF_HOME \$HF_DATASETS_CACHE
nohup bash -c "
  set -euo pipefail
  \$PY -u $DS_ROOT/scripts/fetch_open_mix.py --out-dir \$OUT/raw \\
    --fineweb-docs 9000000 --fineweb-files 14 \\
    --wiki-en-docs 400000 --wiki-zh-docs 200000 \\
    --math-docs 200000 --c4-docs 2000000 --code-docs 800000 \\
    --cc-warcs 40
  \$PY -u $DS_ROOT/scripts/tokenize_corpus.py \\
    --out-dir \$OUT \\
    --source jsonl-dir --jsonl-dir \$OUT/raw/mix \\
    --train-tokens 12000000000 --val-tokens 50000000
" > $DS_LOGS/fetch_tokenize_l2.log 2>&1 &
echo \$! > $DS_LOGS/fetch_tokenize_l2.pid
echo "fetch_tokenize_l2 pid=\$(cat $DS_LOGS/fetch_tokenize_l2.pid)"
EOF
echo "L2 fetch started on $HOST; tail $DS_LOGS/fetch_tokenize_l2.log"
