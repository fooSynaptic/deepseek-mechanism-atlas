# DSA（DeepSeek Sparse Attention）逻辑详解

> [← 中文导读](../README.md) · [← 仓库首页（EN）](../README.md) · [← 系列目录](./README.md) · **[DSA 梗概](../versions/dsa-sparse-attention.md)** · [Index Share 逻辑](./index-share-logic.md) · [V3.2 梗概](../versions/v3-2.md) · [演进总览 §3.6](../reports/deepseek-version-lineage-20260625.md#36-deepseek-v32--v32-exp) · [Raschka §4](../reports/raschka-technical-deepseek-v3-v32-highlights.md#与本地文档映射)

---

## 1. 定位：V3.2 相对 V3.1 的唯一架构改动

DeepSeek-V3.2（2025-12 正式版；2025-09 先有 V3.2-Exp）在 **671B MoE + MLA** 基座上，相对 V3.1-Terminus **只改了一处**：用 **DSA** 替换稠密长上下文注意力。

| 维度 | V3.1-T | V3.2 |
|------|--------|------|
| 参数量 / 激活 | 671B / 37B | 同左 |
| 上下文 | 128K | 128K |
| 注意力 | 稠密 MLA（Hybrid Prefill/Decode） | **稀疏 MLA**：indexer 选 top-$k$ 再算 |
| 公开 benchmark | 基线 | 与 V3.1-T **基本持平** |

V3.2-Exp 与 V3.2 **权重架构相同**；Exp 验证「稀疏不损精度」，正式版为续训 + 后训练成品。

> **梗概**：[docs/versions/v3-2.md](../versions/v3-2.md)

---

## 2. 问题：稠密注意力的 $O(L^2)$ 墙

标准多头注意力对长度为 $L$ 的序列，QK 点积与 softmax 归一化的有效复杂度为 **$O(L^2)$**（相对上下文长度二次增长）。在 128K 长上下文下：

- **Prefill**：算量与 KV 写入随 $L$ 急剧上升；
- **Decode**：每步虽只增 1 token，但需 attend 全长 cache，$T=L$ 时 attention 项为 **$O(L)$** per step，累积仍可观；
- **KV Cache 显存**：MLA 已压缩 KV，但全长 latent 仍随 $L$ 线性涨。

DSA 的目标：**在几乎不损精度的前提下，把「对谁做 attention」变成稀疏问题**，从而把主 attention 降到 **$O(Lk)$**，$k \ll L$。

---

## 3. 机制：两阶段稀疏注意力

DSA 每层（在 MLA 路径上）拆成 **索引** 与 **核心注意力** 两阶段：

<img src="./diagrams/dsa-pipeline.svg" alt="DSA 两阶段：Lightning Indexer → Top-k → Core MLA；Indexer-Cache 与 Latent-Cache" width="920"/>

[直接打开 SVG](./diagrams/dsa-pipeline.svg)

### 3.1 Lightning Indexer（廉价全扫）

> **专题**：[lightning-indexer.md](./lightning-indexer.md)（打分公式 $I_{t,s}$、Indexer-Cache、训练对齐、DeepGEMM）

- 用 **极低 head 维** 的点积：**当前** indexer query $q_t$ 对 **每条历史** token $s$ 的 indexer key $k_s$（Indexer-Cache 第 $s$ 行）打分，得 $I_{t,s}$（见 [lightning-indexer walkthrough](./lightning-indexer.md#decode-forward-walkthrough)）。
- 复杂度仍是 **$O(L^2)$** 量级（每层、每 query 要扫全长），但常数极小，远轻于完整 MLA attention。
- 输出：每个历史位置的重要性分数，用于后续 top-$k$ 选择。

### 3.2 Top-$k$ Selector

- 从全长中选出 **$k=2048$** 个最重要的 latent entry（论文与梗概中的典型值）。
- 得到稀疏 **index 集合** $I$，$|I|=k$。

### 3.3 Core MLA Attention（稀疏主算子）

- **仅对** $I$ 中的 entry 做标准 MLA attention（$O(Lk)$，$k$ 固定）。
- 这是精度敏感的主路径；稀疏模式由 indexer 决定「看哪里」。

> **逻辑链**：Indexer 贵但轻 → 主 attention 贵但只算 $k$ 个位置 → 总长 $L$ 很大时总收益显著。

---

## 4. 异构 KV Cache：Indexer-Cache 与 Latent-Cache

DSA 不仅改算子，还 **显式分裂 KV 存储**，为后续 offload 与 Index Share 提供结构基础（见你截图中的「Index Cache / Latent Cache」表述；正式文档称 **Indexer-Cache** / **Latent-Cache**）。

| Cache 类型 | 存什么 | 典型占比（ESS 论文） | 访问模式 | GPU 常驻 |
|------------|--------|---------------------|----------|----------|
| **Indexer-Cache** | indexer 所需 KV，用于打分、选 top-$k$ | ~16.8% | **每步全量参与** indexer 计算 | **是** |
| **Latent-Cache** | 核心 MLA attention 的 latent KV | ~83.2% | 仅 **被选中的 top-$k$ entry** 进入主 attention | 可 offload（[ESS](../versions/ess-latent-cache-offload.md)） |

**设计含义**：

1. **Indexer-Cache 必须留在 GPU**：每 decode step 都要对全长跑 indexer，offload 会拖垮延迟。
2. **Latent-Cache 可分层**：主 attention 只 touch $k$ 个 entry → [ESS](../versions/ess-latent-cache-offload.md) 等方案可把 **低频、未选中** 的 latent 放到 CPU，按稀疏访问拉回。
3. **Index Share 的「Index」**：后文 infra 补丁复用的是 **top-$k$ index 集合**（及 F 层算出的 indexer 结果），与 Indexer-Cache 常驻 GPU 并不矛盾——省的是 **重复跑 indexer 算子**，不是省 Indexer-Cache 本身。

> **延伸**：[Index Share 逻辑](./index-share-logic.md) · [ESS 详解](../versions/ess-latent-cache-offload.md)

---

## 5. 与 V3.1 Hybrid 推理的关系

V3.1 引入 **Hybrid**：Prefill 用 MHA 形态、Decode 用 MQA 形态，降低 decode KV 带宽（MHA / GQA / MQA 对照见 [V3.1 梗概 §MLA 模式切换](../versions/v3-1.md#mla-模式切换terminus-起)）。DSA 在 V3.2 上 **叠加稀疏选择**，不是推翻 MLA：

- MLA 仍负责 **低秩 KV 压缩**；
- DSA 在 MLA latent 序列上增加 **「选哪些位置参与 attention」**；
- 推理栈需 **自定义 sparse attention 路径**（vLLM day-0、FlashMLA paged kernel 等）。

---

## 6. 推理 infra 关注点

| 组件 | 作用 |
|------|------|
| **DeepGEMM** | indexer logit kernel（含 paged 版） |
| **FlashMLA** | sparse attention 的 paged kernel |
| **Index Share** | 跨层复用 index，减 indexer 次数 |
| **[ESS](../versions/ess-latent-cache-offload.md)** | Latent-Cache CPU offload，与 Index Share **正交可叠加** |

---

## 7. 上下游

| 方向 | 关系 |
|------|------|
| 上游 | [V3.1-Terminus](../versions/v3-1.md)：128K + Hybrid，无 DSA |
| 同架构 | V3.2-Exp → V3.2 正式版 |
| 下游 infra | [Index Share](./index-share-logic.md)、[ESS](../versions/ess-latent-cache-offload.md) |
| 下游算法 | [CSA/HCA 混合压缩注意力](../versions/csa-hca-mixed-attention.md) · [V4 梗概](../versions/v4.md)：压缩 + 稀疏的下一代 |
| 正交 | [Engram 条件记忆](../material/papers/engram/engram-series-overview.md)：n-gram 查表稀疏轴 |

---

## 8. 参考

- 论文：[arXiv:2512.02556](https://arxiv.org/pdf/2512.02556)
- 代码：[deepseek-ai/DeepSeek-V3.2](https://github.com/deepseek-ai/DeepSeek-V3.2)
- 一页梗概：[v3-2.md](../versions/v3-2.md)
- 系列目录：[./README.md](./README.md)
