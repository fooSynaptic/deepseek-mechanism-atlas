# 开发索引（非阅读入口）

读者请从 **[版本演进总览](./reports/deepseek-version-lineage-20260625.md)** 进入；本页仅供维护源路径与构建。

---

## 源稿与成书

| 源 | 书中章节 |
|----|----------|
| [reports/deepseek-version-lineage-20260625.md](./reports/deepseek-version-lineage-20260625.md) | `01-总览/01-版本演进总览.md` |
| `docs/` 下其余 `versions/`、`reports/` | 见 `build_book.py` 内 `CHAPTER_MAP` |

```bash
python3 scripts/svg/check_svgs.py   # 改 SVG 后须 exit 0
python3 《ds-技术报告》/build_book.py
```

<details>
<summary>全量路径表（编辑时查阅）</summary>

### 子项目

| 项目 | 路径 |
|------|------|
| Engram | [docs/engram/](engram/) |
| RL 笔记 | [docs/rl/](rl/) |
| DSA / Index Share | [docs/dsa/](dsa/) |

### 报告与版本

| 文档 | 路径 |
|------|------|
| 版本梗概 | [versions/README.md](./versions/README.md) |
| 报告目录 | [reports/README.md](./reports/README.md) |
| 算法 / infra / MoE 三线 | [algorithm](./reports/deepseek-algorithm-line.md) · [infra](./reports/deepseek-infra-line.md) · [moe](./reports/deepseek-moe-line.md) |
| 补充材料 | [material/](./material/README.md) |

</details>
