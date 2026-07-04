# DeepSeek 基础设施线：MLA KV → 异构 Cache → Index Share → ESS → V4 HiSparse

> [← 中文导读](../README.md) · [← 仓库首页（EN）](../../README.md) · 更新：2026-06-27
> [← 演进总览 §1](deepseek-version-lineage-20260625.md#1-总览) · [算法线导读](deepseek-algorithm-line.md) · **[MoE 线导读](deepseek-moe-line.md)** · [版本梗概索引](../versions/README.md) · [《ds-技术报告》读本](../../《ds-技术报告》/README.md)

<a id="infra-line"></a>

V3 发布之后，DeepSeek 推理侧 **KV cache 与 offload** 的演进可概括为（与 [演进总览 §1](deepseek-version-lineage-20260625.md#1-总览) 正文内联一致）：

**[标准 MLA KV cache](../versions/mla-latent-attention.md) → [Indexer/Latent 异构 cache](../versions/dsa-sparse-attention.md#异构-kv-cache) → [Index Share](../versions/index-share.md) → [ESS offload](../versions/ess-latent-cache-offload.md) → [V4 异构 KV + HiSparse](../versions/v4.md#推理-infra-关注点)**

本文是这条 **基础设施线** 的专题导读与双向跳转 hub；算法侧演进见 [算法线导读](deepseek-algorithm-line.md)，全景时间线见 [版本演进总览](deepseek-version-lineage-20260625.md)。

---

## 1. 演进链

| 阶段 | 核心机制 | 适用版本 | 本地文档 | 论文 / 归属 |
|------|----------|----------|----------|-------------|
| **① 同质 MLA KV** | 单类 latent cache；整条序列同格式 | V2/V3/R1/V3.1 | [MLA 低秩注意力](../versions/mla-latent-attention.md) · [演进 §5.1](deepseek-version-lineage-20260625.md#51-v3--v31标准-mla-latent-cache) | MLA [2405.04434](https://arxiv.org/abs/2405.04434) |
| **② Indexer/Latent 异构** | DSA 把 cache 拆成 Indexer-Cache + Latent-Cache | V3.2 | [DSA稀疏注意力§异构 KV](../versions/dsa-sparse-attention.md#异构-kv-cache) · [DSA 逻辑详解](../dsa/dsa-logic.md) | DSA [2512.02556](https://arxiv.org/abs/2512.02556) |
| **③ Index Share** | 跨层复用 top-$k$ index；减 indexer 重复计算 | V3.2 / GLM-5（**纯 infra**） | [Index Share 梗概](../versions/index-share.md) · [Index Share 逻辑详解](../dsa/index-share-logic.md) | [2603.12201](https://arxiv.org/abs/2603.12201)（清华 + 智谱） |
| **④ ESS offload** | Latent-Cache → CPU；Indexer 常驻 GPU；稀疏 prefetch | V3.2 / GLM-5（**纯 infra**） | [ESS Latent offload](../versions/ess-latent-cache-offload.md) · [ESS 论文梗概](../versions/ess-paper-highlights.md) | [2512.10576](https://arxiv.org/abs/2512.10576)（百度百舸） |
| **⑤ V4 异构 KV + HiSparse** | CSA/HCA/SWA/Indexer/tail 多类 cache；C4 offload + 磁盘 prefix | V4 | [CSA/HCA 算法](../versions/csa-hca-mixed-attention.md) · [DeepSeek-V4 梗概§推理 infra](../versions/v4.md#推理-infra-关注点) · [KV layout](../versions/v4-kv-layout.md) · [HiSparse](../versions/v4-hisparse.md) · [磁盘 prefix](../versions/v4-disk-prefix-cache.md) | [2606.19348](https://arxiv.org/abs/2606.19348) |
| **⑥ 投机解码 / DSpark** | MTP 原生 + DSpark 线上；**唯一专文** | V4 Flash/Pro 预览引擎 | [投机解码与 DSpark](../versions/dspark-speculative-decoding.md) | [DeepSpec](https://github.com/deepseek-ai/DeepSpec) |

> **③ 与 ④ 并列**：二者都依赖 **② 异构 cache**，分别优化 **indexer 算力** 与 **Latent 显存**；可 **[同开](deepseek-version-lineage-20260625.md#4-index-shareindexcache)**。
> **⑥ 与 ①–⑤ 正交**：DSpark 优化 **decode 步吞吐**，不改变 KV 布局。

---

## 2. 阅读顺序

1. [MLA 低秩注意力](../versions/mla-latent-attention.md) — 理解 **同质 latent KV** 基线
2. [DSA 梗概 §异构 KV](../versions/dsa-sparse-attention.md#异构-kv-cache) — Indexer/Latent **分裂**
3. [Index Share 梗概](../versions/index-share.md) → [逻辑详解](../dsa/index-share-logic.md) — 跨层 index 复用
4. [ESS 概念](../versions/ess-latent-cache-offload.md) → [论文梗概](../versions/ess-paper-highlights.md) — Latent offload
5. [V4 §推理 infra](../versions/v4.md#推理-infra-关注点) — 异构 cache 总览
 - [KV layout §3.5.1](../versions/v4-kv-layout.md) · [HiSparse](../versions/v4-hisparse.md) · [磁盘 Prefix Cache §3.5.2](../versions/v4-disk-prefix-cache.md)（[演进总览 §5.3](deepseek-version-lineage-20260625.md#53-v4异构-kv--hisparse--磁盘-prefix-cache)）
6. [投机解码与 DSpark](../versions/dspark-speculative-decoding.md) — **唯一入口**（MTP、自测、DSpark、MTP-1）

**对照表**：[演进总览 §5.4 三代 offload 对比](deepseek-version-lineage-20260625.md#54-三代-offload-对比)

---

## 3. 节点间关系

| 边 | 关系 |
|----|------|
| ① → ② | DSA **算法改动** 使 cache 天然分为 Indexer / Latent 两类 |
| ② → ③ | Index Share **只优化 indexer 路径**；不改 Latent 布局 |
| ② → ④ | ESS **只 offload Latent-Cache**；Indexer 必须 GPU 常驻 |
| ③ ⊥ ④ | **正交**：一个省算、一个省显存；V3.2 上可叠加 |
| ④ → ⑤ | V4 **非 ESS 简单放大**；围绕 CSA/HCA/SWA **重做** 内存层级 |

---

## 4. 与算法线的交叉

| infra 阶段 | 依赖的算法组件 | 文档 |
|------------|----------------|------|
| ② 异构 cache | DSA（Lightning Indexer + Core MLA） | [算法线 §②](deepseek-algorithm-line.md#1-演进链attention--残差) |
| ⑤ V4 HiSparse | CSA/HCA 压缩 entry | [CSA / HCA](../versions/csa-hca-mixed-attention.md) · [算法线 §③](deepseek-algorithm-line.md#1-演进链attention--残差) |

算法线完整导读见 [算法线导读](deepseek-algorithm-line.md)。**MoE 线**见 [MoE 线导读](deepseek-moe-line.md)。

---

## 5. 反向引用

| 节点文档 | 文首应含 |
|----------|----------|
| [MLA 低秩注意力](../versions/mla-latent-attention.md) | `[← 基础设施线导读](deepseek-infra-line.md)` |
| [DSA 稀疏注意力](../versions/dsa-sparse-attention.md) | 同上 + 下游 Index Share / ESS |
| [Index Share 梗概](../versions/index-share.md) | 同上 + 并列 ESS |
| [ESS Latent offload](../versions/ess-latent-cache-offload.md) | 同上 + 并列 Index Share |
| [DeepSeek-V4](../versions/v4.md) | 同上 + 说明与 V3.2 ESS 差异 |
| [V4 KV Layout](../versions/v4-kv-layout.md) · [V4 HiSparse](../versions/v4-hisparse.md) · [V4 磁盘 Prefix Cache](../versions/v4-disk-prefix-cache.md) | V4 infra **三专题**；文首链回 [§5.3](deepseek-version-lineage-20260625.md#53-v4异构-kv--hisparse--磁盘-prefix-cache) |
| [投机解码与 DSpark](../versions/dspark-speculative-decoding.md) | 投机解码 / DSpark **唯一专文** |

维护约定见 [DeepSeek 版本演进线文档引用约定](../material/meta/deepseek-version-lines-crossrefs.md)。
