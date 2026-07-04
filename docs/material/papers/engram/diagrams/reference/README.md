# Engram 图示参考（SVG 数学排版）

**标杆图**：[engram-01d-multi-head-hash.svg](../engram-01d-multi-head-hash.svg)（Step 3 多头哈希）

正文 md 公式用 `$...$`；**图示内公式**用 SVG `<tspan>`（见仓库 `docs/material/meta/svg-diagram-math.md`）。

## 生成器位置

所有 Python 生成器在 `scripts/svg/engram/`；**本目录仅保留 `.svg` 成品**与 TikZ 备选源（`.tex` / `.sh`）。

| 图 | 生成器 |
|----|--------|
| 01d 多头哈希 | `scripts/svg/engram/gen_engram_01d_svg.py` |

数学辅助库：`scripts/svg/engram/reference/svg_math_helpers.py`

## 改图流程

```bash
python3 scripts/svg/engram/gen_engram_XX_svg.py
python3 scripts/svg/check_svgs.py
```

禁止手改成品 `.svg`。

## TikZ 备选

[engram-01d-multi-head-hash.tex](./engram-01d-multi-head-hash.tex) + [build_engram_01d.sh](./build_engram_01d.sh)（需 `pdflatex` + `pdf2svg`；失败时回退 Python 生成器）。
