# V4 里的 SWA是什么？

[← 返回 CSA/HCA 一句话](../../04-版本代际/05-CSA-HCA混合压缩注意力.md#一句话) · [§4 异构 KV](../../04-版本代际/05-CSA-HCA混合压缩注意力.md#v4-mixed-attention) · [KV Layout §State](../05-V4-KV-Layout.md#state-cache) · [磁盘 Prefix §SWA 三档](../07-V4-磁盘Prefix-Cache.md#swa-三档策略论文-352) · [答疑目录](../../01-总览/qa/README.md)

---

## 一句话

**SWA** 保存最近约 $n_{\text{win}}$个 token 的 **精确、未压缩** K/V，给当前 query 提供 **局部因果邻域**；与 CSA/HCA **压缩 entry 分池**（State cache），eviction 与 **磁盘 prefix 策略** 均 **独立**——部署上 SWA 常常是 **per-token KV 体积大头**。

<a id="行业链"></a>

## 0. 四模块表里的 sliding window 指什么

[Raschka 表 8-1](../../08-外部解读/02-Raschka全文解析.md#表-8-1-transformer-模块演进) 与 [演进总览 §1.2](../../01-总览/01-版本演进总览.md#四模块演进) 的 **Attention 行业链** 中，`sliding window` 指 **Sliding Window Attention（SWA，滑动窗口注意力）**：

| 维度 | 说明 |
|------|------|
| **机制** | 每个 query 只 attend **最近 $W$ 个** token 的 K/V（固定局部窗口），复杂度随序列长度 **线性** 而非 $O(L^2)$ |
| **行业语境** | Mistral 等长上下文模型常用此路数，介于 **GQA**（减 KV 头）与 **MLA**（低秩 latent 压缩 KV）之间 |
| **DeepSeek 落点** | V1–V3 主线走 **MLA → DSA → CSA/HCA**；**V4** 把 SWA 作为 **五类 KV 之一** 与压缩 entry **分池**（[KV Layout §State](../05-V4-KV-Layout.md#state-cache)），保证最近邻域 **不丢 token 级精度** |

即：表里的 sliding window 是 **行业演进节点**；DeepSeek 在 **V4** 才将其 **显式工程化** 进异构 cache，而非 V2 引入 MLA 时的同代产物。

---

## 1. 在 V4 五类 KV 里的角色

| 维度 | **SWA** | **CSA / HCA** |
|------|---------|----------------|
| 表示 | **每 token 一条** 精确 K/V | **按块压缩** 后的 entry（4:1 或 128:1） |
| 覆盖范围 | 最近 **窗口内** 局部 | 远距 **稀疏**（CSA）或 **全局摘要**（HCA） |
| 存放池 | **State cache**（per-request 块） | **Classical KV cache**（对齐 $\mathrm{lcm}(4,128)$） |
| 是否可落盘共享 | 体积大、策略复杂（见 §3） | CSA/HCA **压缩块易落盘** |

CSA/HCA 把历史 **压短**；SWA 保证 **紧邻上下文** 不丢精度——二者 **互补**，不是替代关系。

---

## 2. 为何不能只用压缩 entry

块压缩（4:1 / 128:1）会 **损失 token 级精度**。最近几十个 token 对语法、指代、工具调用格式等 **极敏感**；SWA 用 **滑动窗口** 保留这部分 **dense 局部**，与 CSA top-$k$、HCA dense 摘要 **[并行参与](../../04-版本代际/05-CSA-HCA混合压缩注意力.md#v4-mixed-attention)** 同一层 attention 融合。

---

## 3. 推理 infra：体积与 prefix 策略

| 现象 | 说明 |
|------|------|
| **体积** | Together 早期：全量 SWA 时 per-token KV 可 **高于** 仅 MLA 路径；瓶颈常在 **SWA state** 而非 CSA/HCA 本体（[HiSparse 专文](../06-V4-HiSparse.md)） |
| **Eviction** | 与 C4 inactive offload **正交**；可只保留 **高复用** SWA 检查点 |
| **磁盘 Prefix** | 论文 §3.5.2：**Full / Periodic Checkpointing / Zero** 三档——在 **存储带宽 vs 命中重算** 间取舍（[专文](../07-V4-磁盘Prefix-Cache.md#swa-三档策略论文-352)） |

---

## 4. 与 V3.2 DSA 的对比

| | **V3.2** | **V4 SWA** |
|--|----------|------------|
| 局部性 | 主要靠 **顺序滑动 + indexer top-$k$** | **显式 SWA state** + CSA/HCA 压缩 |
| Cache 类型 | Indexer-Cache + Latent-Cache | 五类异构对象之一 |

---

## 5. 相关阅读

| 文档 | 内容 |
|------|------|
| [V4 KV Layout](../05-V4-KV-Layout.md) | Classical vs State 双池；SWA 进 State |
| [V4 磁盘 Prefix Cache](../07-V4-磁盘Prefix-Cache.md) | SWA 三档落盘策略 |
| [V4 HiSparse](../06-V4-HiSparse.md) | SWA 与 C4 offload 的容量权衡 |
