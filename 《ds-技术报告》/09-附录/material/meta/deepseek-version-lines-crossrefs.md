---
description: DeepSeek 算法线 + 基础设施线 + MoE 线文档双向引用约定
alwaysApply: true
---

# DeepSeek 版本演进线文档引用约定

演进总览 [deepseek-version-lineage-20260625.md](../../../01-总览/01-版本演进总览.md) §1 定义三条主线：

| 线 | 链条 |
|----|------|
| **算法线** | MLA → DSA 稀疏注意力 → CSA/HCA 混合压缩注意力 + mHC |
| **基础设施线** | 标准 MLA KV cache → Indexer/Latent 异构 cache → Index Share → ESS offload → V4 异构 KV + HiSparse |
| **MoE 线** | 稠密 FFN → DeepSeekMoE → aux-loss-free 路由 + $L_{\mathrm{Bal}}$ → Hash MoE + FP4 |

凡新建或修改相关 Wiki / 梗概 / 技术报告，须遵守下列 **双向引用** 规则。

---

## 算法线

### Hub

| 角色 | 路径 |
|------|------|
| **导读（源）** | [docs/reports/deepseek-algorithm-line.md](../../../01-总览/05-算法线导读.md) |
| **书中副本** | [《ds-技术报告》/01-总览/05-算法线导读.md](../../../01-总览/05-算法线导读.md) |

### 节点

| 阶段 | 源文件 |
|------|--------|
| MLA | [mla-latent-attention.md](../../../02-基座架构/02-MLA低秩注意力.md) |
| DSA | [dsa-sparse-attention.md](../../../05-DSA稀疏注意力/02-DSA梗概.md) · [../../../dsa/dsa-logic.md](../../../dsa/dsa-logic.md) |
| CSA/HCA | [csa-hca-mixed-attention.md](../../../04-版本代际/05-CSA-HCA混合压缩注意力.md) · [v4.md](../../../04-版本代际/03-V4.md) |
| mHC | [mhc-manifold-hyper-connections.md](../../../04-版本代际/04-mHC流形约束超连接.md) |

### 节点文首

`[← 算法线导读](../../../01-总览/05-算法线导读.md)`（相对路径按文件位置调整）

---

## 基础设施线

### Hub

| 角色 | 路径 |
|------|------|
| **导读（源）** | [docs/reports/deepseek-infra-line.md](../../../01-总览/06-基础设施线导读.md) |
| **书中副本** | [《ds-技术报告》/01-总览/06-基础设施线导读.md](../../../01-总览/06-基础设施线导读.md) |

### 节点

| 阶段 | 源文件 |
|------|--------|
| ① 同质 MLA KV | [mla-latent-attention.md](../../../02-基座架构/02-MLA低秩注意力.md) |
| ② Indexer/Latent 异构 | [dsa-sparse-attention.md](../../../05-DSA稀疏注意力/02-DSA梗概.md)（§异构 KV Cache） |
| ③ Index Share | [index-share.md](../../../05-DSA稀疏注意力/05-Index-Share梗概.md) · [../../../dsa/index-share-logic.md](../../../dsa/index-share-logic.md) |
| ④ ESS offload | [ess-latent-cache-offload.md](../../../06-推理基础设施/01-ESS概念.md) · [ess-paper-highlights.md](../../../06-推理基础设施/02-ESS论文梗概.md) |
| ⑤ V4 HiSparse | [v4.md](../../../04-版本代际/03-V4.md)（§推理 infra 关注点） |

### 节点文首

`[← 基础设施线导读](../../../01-总览/06-基础设施线导读.md)`（与算法线 / MoE 线导读 **并列** 放在 blockquote）

### ③ ⊥ ④ 特殊说明

Index Share 与 ESS 在 V3.2 上 **正交可叠加**；不可写成严格串行依赖。

---

## MoE 线

### Hub

| 角色 | 路径 |
|------|------|
| **导读（源）** | [docs/reports/deepseek-moe-line.md](../../../01-总览/07-MoE线导读.md) |
| **书中副本** | [《ds-技术报告》/01-总览/07-MoE线导读.md](../../../01-总览/07-MoE线导读.md) |

### 节点

| 阶段 | 源文件 |
|------|--------|
| ① 稠密 FFN | [v1.md](../../../04-版本代际/00-V1-LLM.md) |
| ② DeepSeekMoE | [deepseek-moe.md](../../../02-基座架构/05-DeepSeekMoE.md) · [v2.md](../../../04-版本代际/00-V2-MoE与MLA.md) |
| ③ aux-loss-free | [aux-loss-free-moe-routing.md](../../../02-基座架构/03-aux-loss-free-MoE路由.md) · [v3.md](../../../02-基座架构/01-V3基座.md) |
| ④ $L_{\mathrm{Bal}}$ | [moe-sequence-wise-balance-loss.md](../../../02-基座架构/04-序列均衡损失.md) |
| ⑤ Hash MoE + FP4 | [hash-moe-fp4.md](../../../04-版本代际/06-Hash-MoE-FP4.md) · [v4.md](../../../04-版本代际/03-V4.md) |

### 节点文首

`[← MoE 线导读](../../../01-总览/07-MoE线导读.md)`

### ③ + ④ 特殊说明

aux-loss-free 为 **主均衡**；$L_{\mathrm{Bal}}$ 为 **序列内兜底**，互补而非替代。

---

## 演进总览 §1 正文内联（禁止单独 blockquote）

§1 三条链条：**每个术语** 在正文中直接 `[术语](节点文档)` 跳转；**禁止** 在链条下方再用 blockquote 单独挂「导读」行。

MoE 线示例：

`[DeepSeekMoE](../../../02-基座架构/05-DeepSeekMoE.md) → [aux-loss-free 路由](../../../02-基座架构/03-aux-loss-free-MoE路由.md) + [$L_{\mathrm{Bal}}$](../../../02-基座架构/04-序列均衡损失.md) → …`

Hub 文档仍保留专题导读；由节点文首回链，**不**替代演进总览内联。

## 三线 Hub 互链

各 Hub §4（或等价节）须互链另外两条线。

## 索引表联动

以下索引须含 **三条导读** 各一行：

- [docs/versions/README.md](../../../01-总览/02-版本梗概索引.md)
- [docs/reports/README.md](../../../01-总览/03-技术报告索引.md)

## 书中副本

```bash
cd deepseek-everything/《ds-技术报告》 && python3 build_book.py
```

| 源 | 书中 |
|----|------|
| `deepseek-algorithm-line.md` | `01-总览/05-算法线导读.md` |
| `deepseek-infra-line.md` | `01-总览/06-基础设施线导读.md` |
| `deepseek-moe-line.md` | `01-总览/07-MoE线导读.md` |

## 反例

- ❌ 演进总览 §1 链条下方单独 blockquote 挂「导读」
- ❌ 只写 DeepSeekMoE 字符串不链 [deepseek-moe.md](../../../02-基座架构/05-DeepSeekMoE.md) / [aux-loss-free](../../../02-基座架构/03-aux-loss-free-MoE路由.md)
- ❌ 把 $L_{\mathrm{Bal}}$ 写成 aux-loss-free 的前置或替代
- ❌ 把 Hash MoE 与 DSA / ESS 混为同一条线
