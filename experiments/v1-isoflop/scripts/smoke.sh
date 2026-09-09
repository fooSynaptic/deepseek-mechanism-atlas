#!/usr/bin/env bash
# 1-GPU smoke on random packed tokens (~1 min). Local, no SSH.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"

GPU=${SMOKE_GPU:-0}
STEPS=${SMOKE_STEPS:-40}
SMOKE_DATA=$DS_DATA/smoke

echo "[smoke] gpu=$GPU steps=$STEPS root=$DS_ROOT"
mkdir -p "$SMOKE_DATA" "$DS_LOGS" "$DS_RESULTS"
"$DS_PY" "$SCRIPT_DIR/make_smoke_data.py" --out-dir "$SMOKE_DATA"
"$DS_PY" "$SCRIPT_DIR/make_grid.py" --wave a
export PYTHONPATH="$DS_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
CUDA_VISIBLE_DEVICES=$GPU "$DS_PY" "$DS_ROOT/src/train.py" \
  --job-id smoke_s1 \
  --arch-id s1 \
  --c-flops 3e18 \
  --seq-len 512 \
  --no-checkpoint \
  --train-bin "$SMOKE_DATA/train.bin" \
  --val-bin "$SMOKE_DATA/val.bin" \
  --train-bytes "$SMOKE_DATA/train.bytes.bin" \
  --val-bytes "$SMOKE_DATA/val.bytes.bin" \
  --out-dir "$DS_RESULTS/smoke_s1" \
  --smoke-steps "$STEPS" \
  --batch-tokens 4096 \
  --max-lr 1e-3 \
  --micro-seqs 2 \
  --log-every 5 \
  --eval-every-tokens 1e12 \
  --eval-batches 2
echo "[smoke] done -> $DS_RESULTS/smoke_s1"
