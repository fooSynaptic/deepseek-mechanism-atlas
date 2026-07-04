# 维护脚本索引

本仓**阅读材料**（`docs/`、`diagrams/`、`docs/dsa/diagrams/`、`docs/material/**/diagrams/`）只保留 Markdown 与 SVG/图片；**所有 Python 维护脚本**统一放在 `scripts/`。

## 目录

| 路径 | 用途 |
|------|------|
| [svg/](svg/README.md) | DeepSeek / DSA / Engram **SVG 生成器**、`check_svgs.py`、数学 tspan 辅助库 |
| [figures/ess/](figures/ess/README.md) | ESS 论文截图安装、PDF 裁图 |
| [figures/engram/](figures/engram/) | Engram 论文 assets 裁图 |
| [doc_series_gate.sh](doc_series_gate.sh) | 文档系列 CI 门禁（校验 SVG + 成书） |
| [validate_refs.py](validate_refs.py) | 仓内 Markdown 链接零断链校验 |
| [validate_backlinks.py](validate_backlinks.py) | 双向引用校验（顶栏回链 + 成对反向链接） |
| [export_png.py](export_png.py) | 全仓 SVG → `png/` 栅格导出（`cjk_png.py` 注入 Noto CJK 字体；`fonts.conf` 运行时生成，勿提交） |
| [sanitize_sensitive_links.py](sanitize_sensitive_links.py) | 清理私有路径 / 失效 wiki 外链 |

成书入口：[《ds-技术报告》/build_book.py](../《ds-技术报告》/build_book.py)（留在书中目录，非阅读正文）。

## 常用命令

```bash
# 校验全部 SVG + Markdown 嵌入
python3 scripts/svg/check_svgs.py

# 重绘 DeepSeek 主系列图
python3 scripts/svg/gen_deepseek_svgs.py
python3 scripts/svg/gen_dsa_svgs.py

# 重绘 Engram 系列（输出到 docs/material/.engram/diagrams/）
python3 scripts/svg/engram/gen_engram_01d_svg.py

# ESS 论文图：手动截图 inbox
python3 scripts/figures/ess/install_screenshot.py --inbox

# 文档门禁 + 成书
bash scripts/doc_series_gate.sh
```
