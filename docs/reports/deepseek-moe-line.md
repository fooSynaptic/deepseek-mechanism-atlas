# DeepSeek MoE 线：稠密 FFN → DeepSeekMoE → aux-loss-free → Hash MoE

> [← 中文导读](../README.md) · [← 仓库首页（EN）](../../README.md) · 更新：2026-06-27  
> [← 演进总览 §1](deepseek-version-lineage-20260625.md#1-总览) · [算法线导读](deepseek-algorithm-line.md) · [基础设施线导读](deepseek-infra-line.md) · [V1→V3 演进](deepseek-v1-to-v3-lineage.md) · [版本梗概索引](../versions/README.md) · [《ds-技术报告》读本](../../《ds-技术报告》/README.md)

<a id="moe-line"></a>

DeepSeek 在 **FFN / 专家路由** 上的演进可概括为（与 [演进总览 §1](deepseek-version-lineage-20260625.md#1-总览) 正文内联一致）：

**稠密 FFN → [DeepSeekMoE](../versions/deepseek-moe.md) → [aux-loss-free 路由](../versions/aux-loss-free-moe-routing.md) + [$L_{\mathrm{Bal}}$](../versions/moe-sequence-wise-balance-loss.md) → [Hash MoE + FP4](../versions/hash-moe-fp4.md)**

本文是 **MoE 线** 专题导读；Attention 侧见 [算法线导读](deepseek-algorithm-line.md)，KV/offload 见 [基础设施线导读](deepseek-infra-line.md)。

---

## 1. 演进链（FFN / 路由）

| 阶段 | 核心机制 | 定型版本 | 本地文档 | 论文 |
|------|----------|----------|----------|------|
| **① 稠密 FFN** | 全参数 SwiGLU；无稀疏激活 | V1（2024-01） | [v1.md](../versions/v1.md) | [2401.02954](https://arxiv.org/abs/2401.02954) |
| **② DeepSeekMoE** | 细粒度 routed + shared experts；softmax 路由 | V2（2024-05） | [deepseek-moe.md](../versions/deepseek-moe.md) · [v2.md](../versions/v2.md) | [2405.04434](https://arxiv.org/abs/2405.04434) |
| **③ aux-loss-free** | sigmoid affinity + 动态 bias $b_i$；无 aux loss 主均衡 | V3（2024-12）→ V3.2 | [aux-loss-free-moe-routing.md](../versions/aux-loss-free-moe-routing.md) · [v3.md](../versions/v3.md) | [2412.19437](https://arxiv.org/abs/2412.19437) §2.1 |
| **④ $L_{\mathrm{Bal}}$** | 序列内 $f_i P_i$ 互补兜底；极小 $\alpha$ | V3 起 | [moe-sequence-wise-balance-loss.md](../versions/moe-sequence-wise-balance-loss.md) | 同上 Eq. 17–20 |
| **⑤ Hash MoE + FP4** | 前几层 Hash-routed MoE；routed expert FP4 + QAT | V4（2026） | [hash-moe-fp4.md](../versions/hash-moe-fp4.md) · [v4.md](../versions/v4.md) | [2606.19348](https://arxiv.org/abs/2606.19348) |

> **③ 与 ④ 互补**：aux-loss-free $b_i$ 管 **batch 级**主均衡；$L_{\mathrm{Bal}}$ 防 **单序列内** expert 打穿（见 [aux-loss-free §补充](../versions/aux-loss-free-moe-routing.md#补充-sequence-wise-auxiliary-loss-仍在)）。

---

## 2. 阅读顺序

1. [V1 正文](../versions/v1.md) — 稠密基线  
2. [DeepSeekMoE 架构](../versions/deepseek-moe.md) — 细粒度 routed + shared（[Figure 2 优化逻辑](../versions/deepseek-moe.md#optimization-logic) · [Fine-grained vs GShard](../versions/qa/moe-fine-grained-segmentation.md)）  
3. [V2 梗概](../versions/v2.md) — MLA + MoE 版本落地（236B/21B）  
4. [V3 梗概](../versions/v3.md) — 256 / 8 act 旗舰化  
5. [aux-loss-free 路由逻辑](../versions/aux-loss-free-moe-routing.md) → [$L_{\mathrm{Bal}}$ 详解](../versions/moe-sequence-wise-balance-loss.md)  
6. [Hash MoE + FP4](../versions/hash-moe-fp4.md) — Hash 路由与 FP4 量化  
7. [V4 梗概](../versions/v4.md) — 两个规格、Attention、训练与 infra 总览  

前代三代对照：[V1→V3 演进 §3.2 FFN](deepseek-v1-to-v3-lineage.md#32-ffn稠密--moe--大规模-aux-loss-free-moe)

---

## 3. 节点间关系（一句话）

| 边 | 关系 |
|----|------|
| ① → ② | V2 用 **条件计算** 替换稠密 FFN，稀疏激活降训练/推理 FFN 成本 |
| ② → ③ | V3 **扩专家数**（256/8）并改 **sigmoid + bias** 路由，去掉 aux loss 主路径 |
| ③ + ④ | $L_{\mathrm{Bal}}$ **不替代** aux-loss-free，仅序列内安全网 |
| ③ → ⑤ | V4 **继承** DeepSeekMoE 框架；前几层改 **Hash 路由**，并 FP4 量化 routed expert |

---

## 4. 与 Attention / infra 线的交叉

| MoE 阶段 | 正交模块 | 文档 |
|----------|----------|------|
| 全阶段 | Attention / KV | [算法线](deepseek-algorithm-line.md) · [基础设施线](deepseek-infra-line.md) |
| V3.2+ | DSA / Index Share | 不改 MoE 路由权重形状 |
| V4 | mHC 残差 | [mHC](../versions/mhc-manifold-hyper-connections.md) — 子层前后混合，不替代 expert 选择 |

---

## 5. 反向引用（各节点应链回本页）

| 节点文档 | 文首应含 |
|----------|----------|
| [deepseek-moe.md](../versions/deepseek-moe.md) | `[← MoE 线导读](deepseek-moe-line.md)` |
| [v2.md](../versions/v2.md) | `[← MoE 线导读](deepseek-moe-line.md)` |
| [v3.md](../versions/v3.md) | 同上 |
| [aux-loss-free-moe-routing.md](../versions/aux-loss-free-moe-routing.md) | 同上 + 上游 DeepSeekMoE |
| [moe-sequence-wise-balance-loss.md](../versions/moe-sequence-wise-balance-loss.md) | 同上 + 主文档 aux-loss-free |
| [hash-moe-fp4.md](../versions/hash-moe-fp4.md) | 同上 + 上游 aux-loss-free |
| [v4.md](../versions/v4.md) | 同上 + 链 Hash MoE 专文 |

维护约定见 [deepseek-version-lines-crossrefs.mdc](../material/meta/deepseek-version-lines-crossrefs.md)。
