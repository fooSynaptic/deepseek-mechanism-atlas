#!/usr/bin/env bash
# Tokenize a few already-downloaded 100BT parquet shards and append onto
# data_l2/train.bin via a new inode (running jobs keep the old mmap).
# Val.bin is left unchanged. Enough unique → existing watch daemon can launch t2.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
HOST=${1:?usage: append_100bt_unique.sh <host>}
EXTRA_TOKENS=${EXTRA_TOKENS:-800000000}

ssh -o BatchMode=yes "$HOST" bash -s <<EOF
set -euo pipefail
export PYTHONUNBUFFERED=1
ROOT=$DS_ROOT
PY=$DS_VENV/bin/python
SRC=\$ROOT/data_40b/raw/fineweb_100bt_parquet
STAGED=\$ROOT/data_l2/raw/parquet_extra
EXTRA=\$ROOT/data_l2/extra_100bt
mkdir -p \$STAGED \$EXTRA $DS_LOGS
rm -f \$STAGED/*.parquet
# Two ~2GB shards ≈ well over the 0.24B unique gap; skip 10BT *_00000 names.
n=0
for f in 000_00001.parquet 000_00002.parquet 000_00003.parquet; do
  if [[ -f \$SRC/\$f ]]; then
    ln -sfn \$SRC/\$f \$STAGED/\$f
    n=\$((n+1))
  fi
done
echo "staged \$n parquet files from \$SRC"
nohup bash -c "
  set -euo pipefail
  \$PY -u \$ROOT/scripts/tokenize_corpus.py \\
    --out-dir \$EXTRA \\
    --source parquet-dir --parquet-dir \$STAGED \\
    --train-tokens $EXTRA_TOKENS --val-tokens 0
  \$PY -u - <<'PY'
import json, os, shutil
from pathlib import Path
root = Path('$DS_ROOT')
base = root / 'data_l2'
extra = base / 'extra_100bt'
need = 10837208141
old_n = (base / 'train.bin').stat().st_size // 2
add_n = (extra / 'train.bin').stat().st_size // 2
print(f'[append] old={old_n} extra={add_n} -> {old_n+add_n}', flush=True)
def concat_replace(dest, *srcs):
    tmp = dest.with_name(dest.name + '.new')
    with open(tmp, 'wb') as out:
        for s in srcs:
            with open(s, 'rb') as inf:
                shutil.copyfileobj(inf, out, length=16 * 1024 * 1024)
    os.replace(tmp, dest)
concat_replace(base / 'train.bin', base / 'train.bin', extra / 'train.bin')
if (base / 'train.bytes.bin').exists() and (extra / 'train.bytes.bin').exists():
    concat_replace(base / 'train.bytes.bin', base / 'train.bytes.bin', extra / 'train.bytes.bin')
n = (base / 'train.bin').stat().st_size // 2
meta_p = base / 'meta.json'
meta = json.loads(meta_p.read_text()) if meta_p.exists() else {}
meta.update({
    'n_train': n,
    'n_total_written': n + int(meta.get('n_val', 50_000_000)),
    'appended_100bt': add_n,
    'note': 'val.bin unchanged; extra from sample/100BT parquet not in 10BT 14-file set',
})
meta_p.write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps(meta, indent=2), flush=True)
if n < need:
    raise SystemExit(f'still short unique={n} need={need}')
print('[append] unique enough for iso_c3e19_t2', flush=True)
PY
" > $DS_LOGS/append_100bt_unique.log 2>&1 &
echo \$! > $DS_LOGS/append_100bt_unique.pid
echo "append_100bt_unique pid=\$(cat $DS_LOGS/append_100bt_unique.pid)"
EOF
echo "started on $HOST; tail $DS_LOGS/append_100bt_unique.log"
