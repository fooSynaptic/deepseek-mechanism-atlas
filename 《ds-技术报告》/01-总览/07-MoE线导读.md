# DeepSeek MoE 线：稠密 FFN → DeepSeekMoE → aux-loss-free → Hash MoE

> [← 中文导读](../00-前言/02-中文导读.md) · [← 仓库首页（EN）](https://github.com/fooSynaptic/deepseek-tech-notes) · 更新：2026-06-27
> [← 演进总览 §1](01-版本演进总览.md#1-总览) · [算法线导读](05-算法线导读.md) · [基础设施线导读](06-基础设施线导读.md) · [V1→V3 演进](04-V1到V3演进.md) · [版本梗概索引](02-版本梗概索引.md) · [《ds-技术报告》读本](../README.md)

<a id="moe-line"></a>

DeepSeek 在 **FFN / 专家路由** 上的演进可概括为（与 [演进总览 §1](01-版本演进总览.md#1-总览) 正文内联一致）：

**稠密 FFN → [DeepSeekMoE](../02-基座架构/05-DeepSeekMoE.md) → [aux-loss-free 路由](../02-基座架构/03-aux-loss-free-MoE路由.md) + [$L_{\mathrm{Bal}}$](../02-基座架构/04-序列均衡损失.md) → [Hash MoE + FP4](../04-版本代际/06-Hash-MoE-FP4.md)**

本文是 **MoE 线** 专题导读；Attention 侧见 [算法线导读](05-算法线导读.md)，KV/offload 见 [基础设施线导读](06-基础设施线导读.md)。

---

## 1. 演进链

| 阶段 | 核心机制 | 定型版本 | 本地文档 | 论文 |
|------|----------|----------|----------|------|
| **① 稠密 FFN** | 全参数 SwiGLU；无稀疏激活 | V1（2024-01） | [DeepSeek-LLM V1](../04-版本代际/00-V1-LLM.md) | [2401.02954](https://arxiv.org/abs/2401.02954) |
| **② DeepSeekMoE** | 细粒度 routed + shared experts；softmax 路由 | V2（2024-05） | [DeepSeekMoE](../02-基座架构/05-DeepSeekMoE.md) · [DeepSeek-V2](../04-版本代际/00-V2-MoE与MLA.md) | [2405.04434](https://arxiv.org/abs/2405.04434) |
| **③ aux-loss-free** | sigmoid affinity + 动态 bias $b_i$；无 aux loss 主均衡 | V3（2024-12）→ V3.2 | [aux-loss-free MoE 路由](../02-基座架构/03-aux-loss-free-MoE路由.md) · [DeepSeek-V3](../02-基座架构/01-V3基座.md) | [2412.19437](https://arxiv.org/abs/2412.19437) §2.1 |
| **④ $L_{\mathrm{Bal}}$** | 序列内 $f_i P_i$ 互补兜底；极小 $\alpha$ | V3 起 | [序列均衡损失](../02-基座架构/04-序列均衡损失.md) | 同上 Eq. 17–20 |
| **⑤ Hash MoE + FP4** | 前几层 Hash-routed MoE；routed expert FP4 + QAT | V4（2026） | [Hash MoE + FP4](../04-版本代际/06-Hash-MoE-FP4.md) · [DeepSeek-V4](../04-版本代际/03-V4.md) | [2606.19348](https://arxiv.org/abs/2606.19348) |

> **③ 与 ④ 互补**：aux-loss-free $b_i$ 管 **ba[tch 级**主均衡；$L_{\mathrm{Bal}}$ 防 **单序列内** expert 打穿](../02-基座架构/03-aux-loss-free-MoE路由.md#补充-sequence-wise-auxiliary-loss-仍在)。

---

## 2. 阅读顺序

1. [V1 正文](../04-版本代际/00-V1-LLM.md) — 稠密基线
2. [DeepSeekMoE 架构](../02-基座架构/05-DeepSeekMoE.md) — 细粒度 routed + shared（[Figure 2 优化逻辑](../02-基座架构/05-DeepSeekMoE.md#optimization-logic) · [Fine-grained vs GShard](../02-基座架构/qa/moe-fine-grained-segmentation.md)）
3. [V2 梗概](../04-版本代际/00-V2-MoE与MLA.md) — MLA + MoE 版本落地（236B/21B）
4. [V3 梗概](../02-基座架构/01-V3基座.md) — 256 / 8 act 旗舰化
5. [aux-loss-free 路由逻辑](../02-基座架构/03-aux-loss-free-MoE路由.md) → [$L_{\mathrm{Bal}}$ 详解](../02-基座架构/04-序列均衡损失.md)
6. [Hash MoE + FP4](../04-版本代际/06-Hash-MoE-FP4.md) — Hash 路由与 FP4 量化
7. [V4 梗概](../04-版本代际/03-V4.md) — 两个规格、Attention、训练与 infra 总览

前代三代对照：[V1→V3 演进 §3.2 FFN](04-V1到V3演进.md#32-ffn稠密--moe--大规模-aux-loss-free-moe)

---

## 3. 节点间关系

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
| 全阶段 | Attention / KV | [算法线](05-算法线导读.md) · [基础设施线](06-基础设施线导读.md) |
| V3.2+ | DSA / Index Share | 不改 MoE 路由权重形状 |
| V4 | mHC 残差 | [mHC](../04-版本代际/04-mHC流形约束超连接.md) — 子层前后混合，不替代 expert 选择 |

---

## 5. 反向引用

| 节点文档 | 文首应含 |
|----------|----------|
| [DeepSeekMoE](../02-基座架构/05-DeepSeekMoE.md) | `[← MoE 线导读](07-MoE线导读.md)` |
| [DeepSeek-V2](../04-版本代际/00-V2-MoE与MLA.md) | `[← MoE 线导读](07-MoE线导读.md)` |
| [DeepSeek-V3](../02-基座架构/01-V3基座.md) | 同上 |
| [aux-loss-free MoE 路由](../02-基座架构/03-aux-loss-free-MoE路由.md) | 同上 + 上游 DeepSeekMoE |
| [序列均衡损失](../02-基座架构/04-序列均衡损失.md) | 同上 + 主文档 aux-loss-free |
| [Hash MoE + FP4](../04-版本代际/06-Hash-MoE-FP4.md) | 同上 + 上游 aux-loss-free |
| [DeepSeek-V4](../04-版本代际/03-V4.md) | 同上 + 链 Hash MoE 专文 |

维护约定见 [版本演进线文档引用约定](../09-附录/material/meta/deepseek-version-lines-crossrefs.md)。