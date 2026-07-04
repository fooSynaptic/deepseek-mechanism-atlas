#!/usr/bin/env bash
# Doc series CI gate: SVG validation + book build smoke.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> scripts/svg/check_svgs.py"
python3 scripts/svg/check_svgs.py

echo "==> scripts/validate_backlinks.py"
python3 scripts/validate_backlinks.py

echo "==> scripts/validate_refs.py"
python3 scripts/validate_refs.py

echo "==> build_book.py"
python3 《ds-技术报告》/build_book.py

echo "==> spot-check: FP8 chapter in reading order chain"
grep -q "06-V3-FP8动态量化" "《ds-技术报告》/02-基座架构/04-序列均衡损失.md"
if grep -q "## 章节导航" "《ds-技术报告》/02-基座架构/06-V3-FP8动态量化.md"; then
  echo "FAIL: chapter nav footer still present"
  exit 1
fi

echo "OK doc_series_gate"
