# V4 里的 Indexer KV 是什么？

[← 返回 CSA/HCA 一句话](../05-CSA-HCA混合压缩注意力.md#一句话) · [§4 异构 KV](../05-CSA-HCA混合压缩注意力.md#v4-mixed-attention) · [KV Layout §State](../../06-推理基础设施/05-V4-KV-Layout.md#state-cache) · [Lightning Indexer 详解](../../05-DSA稀疏注意力/04-Lightning-Indexer详解.md) · [DSA 梗概](../../05-DSA稀疏注意力/02-DSA梗概.md) · [答疑目录](../../01-总览/qa/README.md)

---

## 一句话

**Indexer KV** 是 V4 **CSA 稀疏路径**上、沿 **V3.2 Lightning Indexer** 一脉的 **轻量 K 向量 cache**：当前 $q_t$ 对 **全长历史** 每条 $k_s$ **廉价打分** → top-$k$ 选出要读的 **C4 压缩 entry**；**维数、池子、是否 offload** 均与 **主 MLA / HCA entry 不同**。

---

## 1. 在 decode 一步做什么

| 步 | 对象 | 动作 |
|----|------|------|
| 1 | 当前 hidden $h_t$ | 现算 indexer query $q_t$ |
| 2 | **Indexer KV** 第 $s$ 行 $k_s$ | 对 **每个历史位置** $s$ 打分 $I_{t,s}$ |
| 3 | Top-$k$ | 得到下标集 $I$ |
| 4 | **Classical 池 C4 entry** | Core attention **只读** $I$ 内压缩块 |

方向始终是 **$q_t \to k_s$**（当前选历史），不是每个历史 token 各自 top-$k$。机制细节见 [Lightning Indexer §1](../../05-DSA稀疏注意力/04-Lightning-Indexer详解.md#1-在-dsa-中的角色)。

---

## 2. 为何单独一类 cache

| 对比 | **Indexer KV** | **CSA C4 entry** | **HCA entry** |
|------|----------------|------------------|---------------|
| 目的 | **选位置** | **被选的远距稀疏** K/V | **全局 dense 摘要** |
| 典型维度 | **窄**（indexer 专用） | 压缩后主 attention | 更重压缩 |
| 读模式 | 全长扫 **打分** | 仅 $I$ 内 **稀疏读** | 短序列 **dense** |
| V3.2 对应 | Indexer-Cache | Latent-Cache | — |

V4 把 indexer 与 C4/HCA **分开记账**，引擎才能 **分池 eviction、分类型 prefix**（[KV layout](../../06-推理基础设施/05-V4-KV-Layout.md)）。

---

## 3. 与 ESS / offload 的关系

| 项 | 说明 |
|----|------|
| **Indexer** | V3.2 ESS：**一般不 offload**（GPU 常驻，先算 index 再决定搬哪些 latent） |
| **V4** | Indexer KV 仍偏 **热路径**；**inactive C4** 的 [HiSparse offload](../../06-推理基础设施/qa/h2d-d2h-pcie-transfer.md) **不替代** indexer 打分（V3.2 链路描述；V4 对象名不同、分工类似） |

---

## 4. 相对纯 DSA 的延续

算法线：**MLA → DSA（indexer + top-$k$）→ CSA（压缩序列上再稀疏）**。Indexer **不是 V4 新发明**，而是 CSA 在 **更短 entry 序列** 上仍需要 **「看谁」** 的阶段；HCA 路径则走 **dense attend 压缩摘要**，不共用同一套 indexer 读法。

---

## 5. 相关阅读

| 文档 | 内容 |
|------|------|
| [Lightning Indexer 详解](../../05-DSA稀疏注意力/04-Lightning-Indexer详解.md) | $q_t$ 对 $k_s$ 打分、$O(L^2)$ 常数极小 |
| [DSA 稀疏注意力](../../05-DSA稀疏注意力/02-DSA梗概.md) | DSA 流水线总览 |
| [CSA / HCA](../05-CSA-HCA混合压缩注意力.md) | CSA 4:1 块 + indexer 在 V4 栈中的位置 |
