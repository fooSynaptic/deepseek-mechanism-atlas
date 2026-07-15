# SVG 生成与校验

生成器输出目录（**本目录不放 .py**）：

| 生成器 | 输出 |
|--------|------|
| `gen_deepseek_svgs.py` 及 `gen_*_svg.py` | [diagrams/](../../diagrams/) · [docs/figures/](../../docs/figures/) |
| `gen_dsa_svgs.py` | [docs/dsa/diagrams/](docs/dsa/diagrams/) |
| [engram/](engram/) 下 `gen_engram_*.py` | [docs/material/papers/engram/diagrams/](../../docs/material/papers/engram/diagrams/) |

## 校验

```bash
python3 scripts/svg/check_svgs.py   # 须 exit 0
```

## 数学排版

- 扩展辅助库：[reference/svg_math_helpers.py](reference/svg_math_helpers.py)（MTP / V3+）
- Engram 基础库：[engram/reference/svg_math_helpers.py](engram/reference/svg_math_helpers.py)
- 约定说明：[reference/README.md](reference/README.md) · [docs/material/meta/svg-diagram-math.md](../../docs/material/meta/svg-diagram-math.md)

## 标杆图

| SVG | 生成器 |
|-----|--------|
| [diagrams/mtp-fusion-scheme.svg](../../diagrams/mtp-fusion-scheme.svg) | `gen_mtp_fusion_scheme_svg.py` |
| [engram-01d-multi-head-hash.svg](../../docs/material/papers/engram/diagrams/engram-01d-multi-head-hash.svg) | `engram/gen_engram_01d_svg.py` |

改图：**只改本目录生成器**，再跑 `check_svgs.py`；禁止手改成品 SVG。

## PNG 导出与 CJK 字体

`export_png.py` 依赖 [cjk_png.py](cjk_png.py) 将 SVG 栅格化为 `png/`。首次导出时会：

1. 按需下载 `NotoSansCJKsc-Regular.otf` 到 `fonts/`（已在 `.gitignore`）
2. **自动生成** `fonts/fonts.conf`（fontconfig 需绝对路径，由脚本按本机仓库根目录写入；**勿手动编辑或提交**）
3. fontconfig 缓存写入 `/tmp/fontconfig-deepseek-mechanism-atlas-png`（不入仓）

```bash
python3 scripts/export_png.py
```
