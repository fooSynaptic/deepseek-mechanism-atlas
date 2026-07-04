# ESS 论文插图维护

阅读正文见 [docs/versions/ess-paper-highlights.md](../../../docs/versions/ess-paper-highlights.md)。

## 手动截图（推荐）

1. 将裁剪好的 `fig-N.png` / `table-N.png` 放入 [docs/figures/ess/paper/screenshots/](../../../docs/figures/ess/paper/screenshots/)
2. 安装到 `paper/`：

```bash
python3 scripts/figures/ess/install_screenshot.py --inbox
# 或单张
python3 scripts/figures/ess/install_screenshot.py fig-2 /path/to/shot.png
```

已登记 `.manual-figures` 的文件不会被 `extract_paper_figures.py` 覆盖。

## PDF 自动裁图（备选）

```bash
python3 scripts/figures/ess/extract_paper_figures.py
```
