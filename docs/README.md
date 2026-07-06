# deepseek-tech-notes · 中文导读

> **[英文首页](../README.md)**

> **丝滑阅读 × 深度拆解 × 前沿跟进** — 非官方 DeepSeek 技术笔记（V1→V4）。**与 DeepSeek 官方无隶属关系**。

📖 **[在线成书（mdBook）](https://fooSynaptic.github.io/deepseek-tech-notes/)** — 与本地 IDE Preview 渲染一致；在线请用 Pages，**勿用 GitHub blob 预览**。

### 推荐阅读

本笔记是**双向引用 wiki**：文首有反向回链，文内有正向深入链接。要发挥这套导航的价值，请用下面两种方式之一——**不要用 GitHub 仓库内的 blob 预览**。

| 方式 | 何时用 | 导航怎么玩 |
|------|--------|------------|
| **IDE Preview**（VS Code / Cursor） | 已 clone 仓库、本地精读或改稿 | 点文首 `←` 回链与文内链接即可跳转；可开**预览分栏**或沿预览历史回溯——**正向 / 反向引用价值最大** |
| **[GitHub Pages（mdBook）](https://fooSynaptic.github.io/deepseek-tech-notes/)** | 在线阅读、无需 clone | 公式、图示与 IDE 一致；用浏览器 **后退 / 前进** 沿阅读路径返回上一篇或再进下一篇，效果与 IDE 里点链接类似 |

**小结**：本地 **IDE Preview** 与 **Pages** 二选一即可；编辑与 PR 仍在本仓库 `docs/` 进行。

### 为何单独建在线 Pages？

本地 **IDE Preview** 与 **GitHub 仓库内 Markdown 预览** 的渲染引擎不同——引用块、行内/块级公式、链接里的 `$...$` 等在 GitHub 上常会错位，在 IDE 里却正常。源稿 `.md` **不为迁就 GitHub Preview 而改写法**；改为用 mdBook + KaTeX 部署 **[在线成书](https://fooSynaptic.github.io/deepseek-tech-notes/)**，与 IDE 阅读体验对齐。在线请点 Pages；改稿、提 PR 仍走本仓库。

> **善意提醒**：正文里的 SVG 插图下方，通常都有 **「图示详情」** 链接——点进去可在新页查看可缩放的原图。不少机制就写在图里的箭头、分区与小字标注里，值得放慢节奏、仔细品读。

> **项目仍在完善中**：梗概补全、书中镜像、链接与图示校验仍在推进。阅读时请以各篇文首的 arXiv / 官方 PDF 为准；发现断链、口径不一致或表述错误，**欢迎提 issue**。

---

## 这个项目在做什么

我从 [DeepSeek V1 技术报告](versions/v1.md) 一路跟到 [**V4**](versions/v4.md)，并把**大部分**主要技术文章里的**机制与细节**拆开写清楚：架构怎么变、训练/推理在优化什么、[版本之间如何衔接](reports/deepseek-version-lineage-20260625.md#2-版本时间线与关系)。

范围包括：

- **DeepSeek 主线**（见 [算法线](reports/deepseek-algorithm-line.md) · [MoE 线](reports/deepseek-moe-line.md)）：[**MLA**](versions/mla-latent-attention.md)、[**DeepSeekMoE**](versions/deepseek-moe.md)、[**aux-loss-free 路由**](versions/aux-loss-free-moe-routing.md)、[**MTP**](versions/v3.md#三mtpmulti-token-prediction)、[**RLVR**](versions/rlvr.md) / [**R1**](versions/r1.md)、[**DSA**](versions/dsa-sparse-attention.md)、[**CSA / HCA**](versions/csa-hca-mixed-attention.md)、[**mHC**](versions/mhc-manifold-hyper-connections.md)、[**Hash MoE**](versions/hash-moe-fp4.md)、[**V4 异构 KV**](versions/v4-kv-layout.md) 等。
- **V4 及衍生的推理技术**（见 [基础设施线](reports/deepseek-infra-line.md)）：如 [**DSpark**](versions/dspark-speculative-decoding.md) 投机解码（半自回归 draft + 置信度调度验证）、[**HiSparse**](versions/v4-hisparse.md)、[**磁盘 Prefix Cache**](versions/v4-disk-prefix-cache.md) 等。
- **叠在 DeepSeek checkpoint 上的衍生工作**——尤其 **AI Infrastructure** 向：
 - [**Index Share / IndexCache**](versions/index-share.md)（清华 + 智谱）：跨层复用 [DSA](versions/dsa-sparse-attention.md) indexer 的 top-$k$ index，纯推理补丁；[逻辑详解](dsa/index-share-logic.md)
 - [**ESS**](versions/ess-latent-cache-offload.md)（百度百舸）：Latent-Cache CPU offload，与 DSA 算法正交；[论文梗概](versions/ess-paper-highlights.md)

---

## 演进

**[版本演进总览](reports/deepseek-version-lineage-20260625.md)** — 全系列唯一主线入口：时间线 + 算法 / 基础设施 / MoE 三线；各版本与 infra 补丁的内链均从此文展开。

<img src="../diagrams/deepseek-version-lineage.svg" alt="DeepSeek 版本时间线：V3 至 V4 算法演进与 Index Share / ESS / DSpark / HiSparse 基础设施补丁" width="920"/>

[图示详情](../diagrams/deepseek-version-lineage.svg) · 与 [演进总览 §1](reports/deepseek-version-lineage-20260625.md#1-总览) 对照阅读

<img src="../diagrams/grpo-vs-ppo.svg" alt="PPO vs GRPO：RLHF 神经 RM + Critic 与 RLVR 验证器 + 组内 baseline 对比" width="920"/>

[图示详情](../diagrams/grpo-vs-ppo.svg) · [RLVR / GRPO](versions/rlvr.md) · [R1](versions/r1.md)

<img src="../diagrams/mtp-fusion-scheme.svg" alt="MTP 融合：主网单步 1 次前向，MTP 链补 draft，无需 K 遍完整前向" width="920"/>

[图示详情](../diagrams/mtp-fusion-scheme.svg) · [DSpark 投机解码](versions/dspark-speculative-decoding.md) · [MTP 融合 scheme](versions/qa/mtp-fusion-scheme.md)

---

## 文章

| 主题 | 一句话 |
|------|--------|
| [**V1**](versions/v1.md) | DeepSeek-LLM 完整中文译文 |
| [**V1 BBPE**](versions/v1-bbpe-tokenizer.md) | Byte-level BPE 词表与预分词 |
| [**V2**](versions/v2.md) | 236B/21B；MLA + DeepSeekMoE 首次引入 |
| [**V3**](versions/v3.md) | 671B MoE + MLA 开源旗舰基座 |
| [**V3 FP8**](versions/v3-fp8-dynamic-quantization.md) | 训练侧 FP8 块级动态量化 |
| [**R1**](versions/r1.md) | V3-Base + RLVR；架构不变 |
| [**RLVR / GRPO**](versions/rlvr.md) | 可验证奖励 + 组内相对优化 |
| [**V3.1**](versions/v3-1.md) | Hybrid 推理，128K |
| [**V3.2**](versions/v3-2.md) | DSA 稀疏注意力 |
| [**DSA**](versions/dsa-sparse-attention.md) | indexer + top-$k$ + Core MLA |
| [**Index Share**](versions/index-share.md) | IndexCache 纯 infra 补丁 |
| [**ESS**](versions/ess-latent-cache-offload.md) · [论文梗概](versions/ess-paper-highlights.md) | Latent-Cache CPU offload |
| [**V4**](versions/v4.md) | CSA + HCA + mHC；1M context |
| [**CSA / HCA**](versions/csa-hca-mixed-attention.md) | 4:1 稀疏 + 128:1 dense 混合压缩注意力 |
| [**mHC**](versions/mhc-manifold-hyper-connections.md) | 双随机流形约束超连接 |
| [**Hash MoE + FP4**](versions/hash-moe-fp4.md) | Hash 路由 + routed expert FP4 |
| [**V4 KV**](versions/v4-kv-layout.md) | Classical + State 双池 |
| [**V4 HiSparse**](versions/v4-hisparse.md) | inactive C4 CPU offload |
| [**V4 磁盘 Prefix**](versions/v4-disk-prefix-cache.md) | CSA/HCA 落盘 + SWA 三档策略 |
| [**DSpark**](versions/dspark-speculative-decoding.md) | V4 投机解码：半自回归 draft + 置信度验证 |
| [**MLA**](versions/mla-latent-attention.md) | latent 压缩 KV |
| [**DeepSeekMoE**](versions/deepseek-moe.md) | 细粒度 routed + shared experts |
| [**MoE 路由**](versions/aux-loss-free-moe-routing.md) | aux-loss-free 动态 bias 负载均衡 |
| [**$L_{\mathrm{Bal}}$**](versions/moe-sequence-wise-balance-loss.md) | 序列内专家均衡损失 |
| [**Hyper-Connections**](versions/hyper-connections.md) | $n$ 路并行残差流；mHC 前置 |

---

<!-- book:omit -->
## 如何新增内容 & 维护成书

**正文源稿在 `docs/`。** [《ds-技术报告》/](../《ds-技术报告》/) 是 **build 生成的书籍镜像**，不要手改其中的 Markdown（会被 `build_book.py` 覆盖）。

新增或移动文档时：

1. **在 `docs/` 下写稿**（如 `docs/versions/`、`docs/dsa/`、`docs/reports/`、`docs/versions/qa/`）。
2. **在 [`《ds-技术报告》/build_book.py`](../《ds-技术报告》/build_book.py) 里登记**：
 - `CHAPTER_MAP` — `docs/...` → 书中章节路径；
 - `READING_ORDER` — 上下章导航顺序；
 - `QA_DESTINATIONS` — 若是答疑页（可镜像到多个书内目录）；
 - `ASSET_MAP` — 仅当新图需要复制进书内目录时。
3. **补导航** — 文首 blockquote 顶栏加 `←` 回链（参考现有篇章）；在相关总览 / 索引里链到新文。
4. **重建并校验**（仓库根目录）：

```bash
python3 《ds-技术报告》/build_book.py
python3 scripts/validate_refs.py
python3 scripts/validate_backlinks.py
```

或一键：`bash scripts/doc_series_gate.sh`。

本地预览 mdBook 站点（需安装 [mdBook](https://rust-lang.github.io/mdBook/)）：

```bash
bash scripts/build_pages.sh
# 打开 mdbook-out/index.html
```

只在 `docs/` 阅读/wiki 模式可不跑 build；要让章节进入 [《ds-技术报告》](../《ds-技术报告》/01-总览/01-版本演进总览.md) 并重写链路与章节导航时，必须执行 `build_book.py`。推送到 `main` 会自动重建 GitHub Pages。

<!-- /book:omit -->

---

## 许可

| 范围 | 许可 |
|------|------|
| 导读、图示、成书读本 | [CC BY 4.0](../LICENSE) |
| `scripts/` | [MIT](../LICENSE-MIT) |
| `docs/engram/` | [Apache 2.0](engram/LICENSE) |
| `docs/material/` 镜像 | 上游 / 原论文许可 |

DeepSeek 论文、权重与官方代码库另有其许可；引用时请以 **arXiv / 官方发布** 为准。
