#!/usr/bin/env bash
# TikZ -> PDF -> SVG for engram-01d (optional path B). Output: ../engram-01d-multi-head-hash.svg
set -euo pipefail
REF="$(cd "$(dirname "$0")" && pwd)"
DIAG="$(dirname "$REF")"
BASE=engram-01d-multi-head-hash
cd "$REF"

if command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode -halt-on-error "$BASE.tex" >/dev/null
  pdflatex -interaction=nonstopmode -halt-on-error "$BASE.tex" >/dev/null
  if command -v pdf2svg >/dev/null 2>&1; then
    pdf2svg "$BASE.pdf" "$DIAG/$BASE.svg"
  elif command -v dvisvgm >/dev/null 2>&1; then
    dvisvgm --pdf "$BASE.pdf" -o "$DIAG/$BASE.svg"
  elif command -v inkscape >/dev/null 2>&1; then
    inkscape "$BASE.pdf" --export-filename="$DIAG/$BASE.svg"
  else
    echo "pdf2svg/dvisvgm/inkscape missing, fallback to gen_engram_01d_svg.py" >&2
    python3 "$DIAG/gen_engram_01d_svg.py"
    exit 0
  fi
else
  echo "pdflatex missing, using gen_engram_01d_svg.py" >&2
  python3 "$DIAG/gen_engram_01d_svg.py"
  exit 0
fi

python3 "$DIAG/check_svgs.py"
echo "OK $DIAG/$BASE.svg"
