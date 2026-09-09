#!/usr/bin/env bash
# Grow data_l2 unique toward L3 floor (≥31.2B, target 40B) from already-downloaded
# FineWeb-Edu sample/100BT parquet (skip shards already appended as 000_00001–03).
# Appends via new inode so any concurrent readers keep the old mmap.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=lib/env.sh
source "$SCRIPT_DIR/lib/env.sh"
HOST=${1:?usage: grow_l3_unique.sh <host>}
# Need ~19.8B more for t0@3e19; target +28.6B → ~40.4B unique.
EXTRA_TOKENS=${EXTRA_TOKENS:-28600000000}

ssh -o BatchMode=yes "$HOST" bash -s <<EOF
set -euo pipefail
export PYTHONUNBUFFERED=1
ROOT=$DS_ROOT
PY=$DS_VENV/bin/python
SRC=\$ROOT/data_40b/raw/fineweb_100bt_parquet
STAGED=\$ROOT/data_l2/raw/parquet_l3
EXTRA=\$ROOT/data_l2/extra_l3
mkdir -p \$STAGED \$EXTRA $DS_LOGS
rm -f \$STAGED/*.parquet
# Skip the three shards already in data_l2 from the 0.8B append.
n=0
for f in \$SRC/*.parquet; do
  base=\$(basename "\$f")
  case "\$base" in
    000_00001.parquet|000_00002.parquet|000_00003.parquet) continue ;;
  esac
  ln -sfn "\$f" "\$STAGED/\$base"
  n=\$((n+1))
done
echo "staged \$n parquet files (skipped 000_00001–03)"
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
extra = base / 'extra_l3'
need_t1 = 18410701089
need_t0 = 31165827334
old_n = (base / 'train.bin').stat().st_size // 2
add_n = (extra / 'train.bin').stat().st_size // 2
print(f'[l3-append] old={old_n} extra={add_n} -> {old_n+add_n}', flush=True)

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
    'appended_l3_100bt': add_n,
    'note': 'val.bin unchanged; L3 growth from sample/100BT parquet excluding prior 0.8B append shards',
})
meta_p.write_text(json.dumps(meta, indent=2) + '\n')
print(json.dumps(meta, indent=2), flush=True)
print(f'[l3-append] unlock t1={n >= need_t1} t0={n >= need_t0}', flush=True)
if n < need_t0:
    raise SystemExit(f'still short for t0 unique={n} need={need_t0}')
print('[l3-append] unique enough for iso_c3e19_t0', flush=True)
PY
" > $DS_LOGS/grow_l3_unique.log 2>&1 &
echo \$! > $DS_LOGS/grow_l3_unique.pid
echo "grow_l3_unique pid=\$(cat $DS_LOGS/grow_l3_unique.pid)"
EOF
echo "started on $HOST; tail $DS_LOGS/grow_l3_unique.log"
