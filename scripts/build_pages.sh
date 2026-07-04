#!/usr/bin/env bash
# Build mdBook site for GitHub Pages (run from repo root).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> build_book.py"
python3 "《ds-技术报告》/build_book.py"

echo "==> gen_mdbook_summary.py"
python3 scripts/gen_mdbook_summary.py

echo "==> mdbook build"
mdbook build

echo "==> fix_mdbook_paths.py"
python3 scripts/fix_mdbook_paths.py

echo ""
echo "Pages 构建完成。本地 smoke test（带 /deepseek-tech-notes/ 前缀）:"
echo "  python3 scripts/mdbook_pages_preview.py"
echo "日常本地阅读请用: bash scripts/serve_local.sh"
echo ""
echo "OK mdbook-out/ ready"
