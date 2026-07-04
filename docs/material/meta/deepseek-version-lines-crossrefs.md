---
description: DeepSeek 算法线 + 基础设施线 + MoE 线文档双向引用约定
alwaysApply: true
---

# DeepSeek 版本演进线文档引用约定

演进总览 [版本演进总览](../../../docs/reports/deepseek-version-lineage-20260625.md) §1 定义三条主线：

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
| **导读（源）** | [算法线导读](../../../docs/reports/deepseek-algorithm-line.md) |
| **书中副本** | [《ds-技术报告》/01-总览/05-算法线导读.md](../../../《ds-技术报告》/01-总览/05-算法线导读.md) |

### 节点

| 阶段 | 源文件 |
|------|--------|
| MLA | [MLA 低秩注意力](../../../docs/versions/mla-latent-attention.md) |
| DSA | [DSA 稀疏注意力](../../../docs/versions/dsa-sparse-attention.md) · [DSA 逻辑详解](../../../dsa/dsa-logic.md) |
| CSA/HCA | [CSA / HCA](../../../docs/versions/csa-hca-mixed-attention.md) · [DeepSeek-V4](../../../docs/versions/v4.md) |
| mHC | [mHC](../../../docs/versions/mhc-manifold-hyper-connections.md) |

### 节点文首

`[← 算法线导读](../../../docs/reports/deepseek-algorithm-line.md)`（相对路径按文件位置调整）

---

## 基础设施线

### Hub

| 角色 | 路径 |
|------|------|
| **导读（源）** | [基础设施线导读](../../../docs/reports/deepseek-infra-line.md) |
| **书中副本** | [《ds-技术报告》/01-总览/06-基础设施线导读.md](../../../《ds-技术报告》/01-总览/06-基础设施线导读.md) |

### 节点

| 阶段 | 源文件 |
|------|--------|
| ① 同质 MLA KV | [MLA 低秩注意力](../../../docs/versions/mla-latent-attention.md) |
| ② Indexer/Latent 异构 | [DSA 稀疏注意力](../../../docs/versions/dsa-sparse-attention.md)（§异构 KV Cache） |
| ③ Index Share | [Index Share 梗概](../../../docs/versions/index-share.md) · [Index Share 逻辑详解](../../../dsa/index-share-logic.md) |
| ④ ESS offload | [ESS Latent offload](../../../docs/versions/ess-latent-cache-offload.md) · [ESS 论文梗概](../../../docs/versions/ess-paper-highlights.md) |
| ⑤ V4 HiSparse | [DeepSeek-V4](../../../docs/versions/v4.md)（§推理 infra 关注点） |

### 节点文首

`[← 基础设施线导读](../../../docs/reports/deepseek-infra-line.md)`（与算法线 / MoE 线导读 **并列** 放在 blockquote）

### ③ ⊥ ④ 特殊说明

Index Share 与 ESS 在 V3.2 上 **正交可叠加**；不可写成严格串行依赖。

---

## MoE 线

### Hub

| 角色 | 路径 |
|------|------|
| **导读（源）** | [MoE 线导读](../../../docs/reports/deepseek-moe-line.md) |
| **书中副本** | [《ds-技术报告》/01-总览/07-MoE线导读.md](../../../《ds-技术报告》/01-总览/07-MoE线导读.md) |

### 节点

| 阶段 | 源文件 |
|------|--------|
| ① 稠密 FFN | [DeepSeek-LLM V1](../../../docs/versions/v1.md) |
| ② DeepSeekMoE | [DeepSeekMoE](../../../docs/versions/deepseek-moe.md) · [DeepSeek-V2](../../../docs/versions/v2.md) |
| ③ aux-loss-free | [aux-loss-free MoE 路由](../../../docs/versions/aux-loss-free-moe-routing.md) · [DeepSeek-V3](../../../docs/versions/v3.md) |
| ④ $L_{\mathrm{Bal}}$ | [序列均衡损失](../../../docs/versions/moe-sequence-wise-balance-loss.md) |
| ⑤ Hash MoE + FP4 | [Hash MoE + FP4](../../../docs/versions/hash-moe-fp4.md) · [DeepSeek-V4](../../../docs/versions/v4.md) |

### 节点文首

`[← MoE 线导读](../../../docs/reports/deepseek-moe-line.md)`

### ③ + ④ 特殊说明

aux-loss-free 为 **主均衡**；$L_{\mathrm{Bal}}$ 为 **序列内兜底**，互补而非替代。

---

## 演进总览 §1 正文内联

§1 三条链条：**每个术语** 在正文中直接 `[术语](节点文档)` 跳转；**禁止** 在链条下方再用 blockquote 单独挂「导读」行。

MoE 线示例：

`[DeepSeekMoE](../../../docs/versions/deepseek-moe.md) → [aux-loss-free 路由](../../../docs/versions/aux-loss-free-moe-routing.md) + [$L_{\mathrm{Bal}}$](../../../docs/versions/moe-sequence-wise-balance-loss.md) → …`

Hub 文档仍保留专题导读；由节点文首回链，**不**替代演进总览内联。

## 三线 Hub 互链

各 Hub §4（或等价节）须互链另外两条线。

## 索引表联动

以下索引须含 **三条导读** 各一行：

- [版本梗概索引](../../../docs/versions/README.md)
- [技术报告索引](../../reports/README.md)

## 书中副本

```bash
cd deepseek-tech-notes/《ds-技术报告》 && python3 build_book.py
```

| 源 | 书中 |
|----|------|
| `deepseek-algorithm-line.md` | `01-总览/05-算法线导读.md` |
| `deepseek-infra-line.md` | `01-总览/06-基础设施线导读.md` |
| `deepseek-moe-line.md` | `01-总览/07-MoE线导读.md` |

## 反例

- ❌ 演进总览 §1 链条下方单独 blockquote 挂「导读」
- ❌ 只写 DeepSeekMoE 字符串不链 [DeepSeekMoE](../../../docs/versions/deepseek-moe.md) / [aux-loss-free](../../../docs/versions/aux-loss-free-moe-routing.md)
- ❌ 把 $L_{\mathrm{Bal}}$ 写成 aux-loss-free 的前置或替代
- ❌ 把 Hash MoE 与 DSA / ESS 混为同一条线
