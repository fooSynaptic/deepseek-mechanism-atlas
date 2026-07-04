# DeepSeek 算法线：MLA → DSA → CSA/HCA + mHC

> [← 中文导读](../README.md) · [← 仓库首页（EN）](../../README.md) · 更新：2026-06-27  
> [← 演进总览 §1](deepseek-version-lineage-20260625.md#1-总览) · [基础设施线导读](deepseek-infra-line.md) · **[MoE 线导读](deepseek-moe-line.md)** · [版本梗概索引](../versions/README.md) · [《ds-技术报告》读本](../../《ds-技术报告》/README.md)

<a id="algorithm-line"></a>

V3 发布之后，DeepSeek 在 **注意力与残差路径** 上的算法演进可概括为（与 [演进总览 §1](deepseek-version-lineage-20260625.md#1-总览) 正文内联一致）：

**[MLA](../versions/mla-latent-attention.md) → [DSA 稀疏注意力](../versions/dsa-sparse-attention.md) → [CSA/HCA 混合压缩注意力](../versions/csa-hca-mixed-attention.md) + [mHC](../versions/mhc-manifold-hyper-connections.md)**

本文是这条 **算法线** 的专题导读与双向跳转 hub；全景时间线（含 infra 线）见 [版本演进总览](deepseek-version-lineage-20260625.md)。

---

## 1. 演进链（Attention + 残差）

| 阶段 | 核心机制 | 首发 / 定型版本 | 本地文档 | 论文 |
|------|----------|-----------------|----------|------|
| **① MLA** | K/V 压入低维 latent 再缓存；Core 仍做多头注意力 | V2（2024-05）→ V3/R1/V3.1 沿用 | [mla-latent-attention.md](../versions/mla-latent-attention.md) | [2405.04434](https://arxiv.org/abs/2405.04434) |
| **② DSA** | Lightning Indexer 选 top-$k$ → 仅对 $k$ 个 latent 做 MLA | V3.2-Exp / V3.2（2025） | [dsa-sparse-attention.md](../versions/dsa-sparse-attention.md) · [dsa-logic.md](../dsa/dsa-logic.md) | [2512.02556](https://arxiv.org/abs/2512.02556) |
| **③ CSA / HCA** | 4:1 / 128:1 压缩 KV + 内嵌 indexer；百万 token | V4（2026） | [csa-hca-mixed-attention.md](../versions/csa-hca-mixed-attention.md) · [v4.md](../versions/v4.md) | [2606.19348](https://arxiv.org/abs/2606.19348) |
| **④ mHC** | 残差 Hyper-Connections → 双随机流形约束 | V4 落地（论文 2025-12） | [mhc-manifold-hyper-connections.md](../versions/mhc-manifold-hyper-connections.md)（含 [§3 双随机流形](../versions/mhc-manifold-hyper-connections.md#3-mhc-核心双随机流形约束)）· [HC 基础](../versions/hyper-connections.md) | [2512.24880](https://arxiv.org/abs/2512.24880) |

> **注意**：mHC 改的是 **残差路径**（与 Attention / KV 正交），在演进总览里与 CSA/HCA **并列** 标注，便于对照 V4 全架构；详见 [mHC §7](../versions/mhc-manifold-hyper-connections.md#7-与-attention--moe-线的关系)。

---

## 2. 阅读顺序（按算法线）

1. [MLA 低秩注意力](../versions/mla-latent-attention.md) — latent KV 压缩基座  
2. [V3.1 Hybrid](../versions/v3-1.md) — Prefill MHA / Decode MQA（DSA 前置）  
3. [DSA 梗概](../versions/dsa-sparse-attention.md) → [逻辑详解](../dsa/dsa-logic.md) → [Lightning Indexer](../dsa/lightning-indexer.md)  
4. [CSA/HCA 混合压缩注意力](../versions/csa-hca-mixed-attention.md) — 4:1 稀疏 + 128:1 dense  
5. [V4 梗概](../versions/v4.md) — 两个规格、MoE、训练与 infra 总览  
6. [Hyper-Connections（HC）](../versions/hyper-connections.md) — 多路残差流基础  
7. [mHC 流形约束超连接](../versions/mhc-manifold-hyper-connections.md) — V4 残差组件  

**外部解读**：[Raschka 要点速读 §3–4 MLA/DSA](raschka-technical-deepseek-v3-v32-highlights.md) · [§8 mHC](raschka-technical-deepseek-v3-v32-highlights.md#结论表-7--8)

---

## 3. 节点间关系（一句话）

| 边 | 关系 |
|----|------|
| MLA → DSA | **MLA 结构不变**；在 latent 序列上加 indexer + top-$k$ 稀疏选择 |
| DSA → CSA/HCA | DSA 的「先选再看」思想延续；V4 先做 **token 块压缩** 再在压缩序列上稀疏 / dense |
| CSA/HCA ⊥ mHC | 前者改 **Attention / KV**；后者改 **残差拓扑**，V4 同期引入 |

---

## 4. 与 infra 线的交叉

完整 **基础设施线** 见 **[基础设施线导读](deepseek-infra-line.md)**。

| 算法阶段 | 常见 infra 补丁 | 文档 |
|----------|-----------------|------|
| DSA | Indexer/Latent 异构 cache、Index Share、ESS | [infra 线 §②–④](deepseek-infra-line.md#1-演进链kv--offload) |
| V4 CSA/HCA | HiSparse、磁盘 prefix cache、异构 KV layout | [infra 线 §⑤](deepseek-infra-line.md#1-演进链kv--offload) · [KV layout](../versions/v4-kv-layout.md) · [HiSparse](../versions/v4-hisparse.md) · [磁盘 prefix](../versions/v4-disk-prefix-cache.md) |

**MoE 线**（FFN / 路由，与 Attention 正交）：见 [MoE 线导读](deepseek-moe-line.md)。

---

## 5. 反向引用（各节点应链回本页）

| 节点文档 | 文首应含 |
|----------|----------|
| [mla-latent-attention.md](../versions/mla-latent-attention.md) | `[← 算法线导读](deepseek-algorithm-line.md)` |
| [dsa-sparse-attention.md](../versions/dsa-sparse-attention.md) | 同上 + 上游 MLA、下游 V4 |
| [csa-hca-mixed-attention.md](../versions/csa-hca-mixed-attention.md) | 同上 + 上游 DSA、下游 infra |
| [v4.md](../versions/v4.md) | 同上 + 链 CSA/HCA 专文 |
| [mhc-manifold-hyper-connections.md](../versions/mhc-manifold-hyper-connections.md) | 同上 + 说明残差路径角色 |

维护约定见 [deepseek-version-lines-crossrefs.md](../material/meta/deepseek-version-lines-crossrefs.md)。
