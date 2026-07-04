# 开发索引

> [← 中文导读](./README.md) · [← 演进总览](./reports/deepseek-version-lineage-20260625.md)

读者请从 **[版本演进总览](./reports/deepseek-version-lineage-20260625.md)** 进入；本页仅供维护源路径与构建。

---

## 源稿与成书

| 源 | 书中章节 |
|----|----------|
| [版本演进总览](./reports/deepseek-version-lineage-20260625.md) | `01-总览/01-版本演进总览.md` |
| `docs/` 下其余 `versions/`、`reports/` | 见 `build_book.py` 内 `CHAPTER_MAP` |

```bash
python3 scripts/svg/check_svgs.py # 改 SVG 后须 exit 0
python3 《ds-技术报告》/build_book.py
```

<details>
<summary>全量路径表（编辑时查阅）</summary>

### 子项目

| 项目 | 路径 |
|------|------|
| Engram | [Engram 系列](engram/) |
| RL 笔记 | [RL / 后训练笔记](rl/) |
| DSA / Index Share | [DeepSeek DSA 与 Index Share 系列](dsa/) |

### 报告与版本

| 文档 | 路径 |
|------|------|
| 版本梗概 | [版本梗概索引](./versions/README.md) |
| 报告目录 | [技术报告索引](./reports/README.md) |
| 算法 / infra / MoE 三线 | [算法线导读](./reports/deepseek-algorithm-line.md) · [基础设施线导读](./reports/deepseek-infra-line.md) · [MoE 线导读](./reports/deepseek-moe-line.md) |
| 补充材料 | [material 镜像](material/README.md) |

</details>
