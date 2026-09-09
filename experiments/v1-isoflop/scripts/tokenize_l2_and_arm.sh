#!/usr/bin/env bash
# Tokenize on-disk L2 jsonl into $DS_ROOT/data_l2 (does not touch 4B data/ or 100BT fetch).
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
HOST=${1:?usage: tokenize_l2_and_arm.sh <host>}

ssh -o BatchMode=yes "$HOST" bash -s <<EOF
set -euo pipefail
export PYTHONUNBUFFERED=1
ROOT=$DS_ROOT
PY=$DS_VENV/bin/python
OUT=\$ROOT/data_l2
MIX=\$OUT/raw/mix
A=\$ROOT/data/raw/mix
mkdir -p \$MIX $DS_LOGS
for f in wiki_en.jsonl openwebmath.jsonl gutenberg.jsonl cc_wet.jsonl; do
  if [[ -f \$A/\$f ]]; then ln -sfn \$A/\$f \$MIX/\$f; fi
done
if [[ -f \$A/wiki_zh.jsonl ]]; then
  rm -f \$MIX/wiki_zh.jsonl
  ln -sfn \$A/wiki_zh.jsonl \$MIX/wiki_zh.jsonl
fi
nohup \$PY -u \$ROOT/scripts/tokenize_corpus.py \\
  --out-dir \$OUT \\
  --source jsonl-dir --jsonl-dir \$MIX \\
  --train-tokens 11000000000 --val-tokens 50000000 \\
  > $DS_LOGS/tokenize_l2.log 2>&1 &
echo \$! > $DS_LOGS/tokenize_l2.pid
echo "tokenize_l2 pid=\$(cat $DS_LOGS/tokenize_l2.pid)"
EOF
echo "tokenize started on $HOST; tail $DS_LOGS/tokenize_l2.log"
