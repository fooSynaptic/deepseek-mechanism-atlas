# ESS 论文图 — 手动截图收件箱

将截好的图按命名放入本目录后，在仓库根执行安装（详见 [scripts/figures/ess/README.md](../../../../../scripts/figures/ess/README.md)）：

```bash
python3 scripts/figures/ess/install_screenshot.py --inbox
```

## 命名

| 文件 | 文档位置 |
|------|----------|
| `fig-1.png` … `fig-9.png` | [ess-paper-highlights.md](../../../../versions/ess-paper-highlights.md) |
| `table-1.png` `table-2.png` | 同上 Table 章节 |

## 截图范围

- **要**：原图 + 论文图注
- **不要**：页眉、§ 小节标题、图注下方正文

已安装的图记入 `../.manual-figures`，不会被 PDF 自动裁图覆盖。
