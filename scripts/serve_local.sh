#!/usr/bin/env bash
# 本地 mdBook 预览（勿跑 fix_mdbook_paths —— 该脚本仅用于 GitHub Pages 构建）。
# 打开 http://127.0.0.1:3000/  （无 /deepseek-tech-notes/ 前缀）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 "《ds-技术报告》/build_book.py"
python3 scripts/gen_mdbook_summary.py
mdbook build

echo ""
echo "本地预览: http://127.0.0.1:3000/"
echo "图示详情 SVG 例: http://127.0.0.1:3000/05-DSA稀疏注意力/figures/ess-dual-cache.svg"
echo "（勿用 /deepseek-tech-notes/ 前缀；Pages 构建请 bash scripts/build_pages.sh）"
echo ""

exec mdbook serve --hostname 127.0.0.1 --port 3000
