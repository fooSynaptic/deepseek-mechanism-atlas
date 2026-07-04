# Prefetch window：不是「CPU 比 GPU 强」，而是 CPU 先点火、GPU 腾出时间窗

[← 返回 §prefetch window](../engram-series-overview.md#缓存访问逻辑一次-decode-的完整路径) · [02c 时序图](../diagrams/engram-02c-cxl-cache-access.svg) · [L1–L3 存储](./cxl-l1-l2-l3-memory-tiers.md) · [CXL vs RDMA](./cxl-vs-rdma-communication-pattern.md) · [答疑目录](./README.md)

---

## 问题

[02c 图](../diagrams/engram-02c-cxl-cache-access.svg) 右侧 **prefetch window** 和左侧绿色 **Step 0/1（CPU）** 连在一起——是不是 **prefetch 放在 CPU 上比 GPU 更强**？为什么？

---

## 结论（先答）

**不是。** Prefetch window **不是一块「放在 CPU 上的内存」**，而是 **一段时间**：

$$
\text{window} = \sum_{i=1}^{k-1} t_{\mathrm{exec}}^{\mathrm{GPU}}(i)
$$

即：Engram 插在 **第 $k$ 层** 时，GPU 先算 **Layer $0 \ldots k-1$** 的这段时间，用来 **并行** 把 Layer $k$ 要的 embedding 从 CXL 拉到 HBM staging。

| 角色 | 做什么 | 「强」在哪 |
|------|--------|------------|
| **CPU（Step 0–1）** | 用 **token IDs** 算 `offsets[]`；SGLang **异步发起** prefetch | **不必等 GPU hidden**；decode 步一开始就点火，**不占 SM** |
| **GPU（窗口主体）** | 算 Layer $0..k-1$（Attn/MoE 等） | 用 **算力时间** 盖住 CXL 读延迟 |
| **CXL / DMA（Step 2）** | L3 → L1 staging（Path B P2P **推荐**） | 与上两行 **overlap**，不是串在 GPU critical path 前面 |

所以图里绿色框多，是因为 **「谁先能知道要读哪」在 CPU/host**；**不是**说 CXL 数据通路 CPU 比 GPU 更快——Step 2 **推荐仍是 CXL→GPU P2P**（`cxl2vram_copy`）。

---

## 0. 基础概念：decode、Layer $k$、hidden、「更早做」

<a id="基础概念-decode-layer-hidden"></a>

### Decode（解码步）是什么

LLM 推理分两段：

| 阶段 | 在干什么 | 一次 forward 处理多少新 token |
|------|----------|-------------------------------|
| **Prefill** | 吃进整段 prompt | 很多 token（整段并行） |
| **Decode** | 已生成回复后，**每步只新来 1 个 token**，再跑一遍模型 | **通常 1 个**新 token / 步 |

**Decode 步** = 每生成一个新 token 时触发的 **一次完整前向**（从 Layer 0 一直算到最后一层，再采样下一个 token）。

[02c 图](../diagrams/engram-02c-cxl-cache-access.svg) 标题 *one Decode step* 画的就是：**这一步** 里 Engram 怎么 prefetch。

### Layer $k$、Layer $k-1$ 是什么

Transformer backbone 是一叠 **相同结构的块**，编号 $0,1,\ldots,L-1$（如 64 层）。

- **Layer $k$**：Engram **插在第 $k$ 块里** 的那一层（论文例：$k=2,15$）。
- **Layer $k-1$**：紧挨在它 **上面** 的一层（下标小 1），要先算完，才能得到喂给 Layer $k$ 的 hidden。

数据流（简化）：

$$
h^{(0)} \xrightarrow{\text{Layer 0}} h^{(1)} \xrightarrow{\text{Layer 1}} h^{(2)} \xrightarrow{\text{Engram@2 + Attn + MoE}} \cdots
$$

上标 $(\ell)$ 表示 **第几层之后** 的表示；$h^{(k-1)}$ 就是 **Layer $k-1$ 算完以后** 的向量。

### Hidden（$h_t$ / $h^{(\ell)}$）是什么

- **$h_t$**：序列里 **位置 $t$** 的 hidden 向量（一个长度为 $d$ 的向量），承载「读到当前 token 为止，模型内部的语境表示」。
- **$h_t^{(\ell)}$**：同一位置 $t$ 在 **第 $\ell$ 层之后** 的 hidden。

Attention、MoE 路由、Engram **门控 $\alpha_t$** 都要看 **hidden**（动态、依赖前文计算）。

对比：

| 量 | 依赖什么 | 何时可知 |
|----|----------|----------|
| n-gram / hash 索引 | **input token IDs** | decode 步 **一开始** 就有 |
| $h_t^{(\ell)}$ | 前面层 Attention/MoE **算出来** | Layer $\ell$ **算完之后** 才有 |

### 「更早做」到底是多早

**更早** = 相对「必须等 Layer $k-1$ 的 hidden 出来之后再决定读什么」。

**Engram（可更早）** — decode 步时间线：

```
时刻 T0  本步新 token 的 ID 已知（整段 input_ids 都在）
         CPU：hash → offsets[]（所有 Engram 层要读哪几行都定了）
         CPU：SGLang 立刻异步向 CXL 发读请求   ← 「更早做」指这里

时刻 T0~T1  GPU 并行算 Layer 0, 1, …, k-1     ← prefetch window（藏 CXL 延迟）
         ‖       CXL 并行把 Layer k 要的 embedding 拉进 staging

时刻 T1  GPU 算 Layer k：staging 里已有 e_t，再 gate(h_t, …)
```

**MoE（不能更早）** — 路由要看 **当前 hidden**：

```
Layer k-1 算完 → 得到 h^{(k-1)} → 才知道选哪个 expert → 才能去读/算
```

所以 MoE 的「该激活谁」**最早**也要等到 $k-1$ 层结束；prefetch **有效窗口 $\approx 0$**（相对 Engram 而言）。

**「更早做」不是**：

- ❌ 比 prefill 更早（prefill 是另一阶段；这里指 **同一 decode 步内** 相对层间依赖更早）
- ❌ CPU 比 GPU 算得快
- ✅ **在 GPU 还没算到 Layer $k-1$ 之前**，仅凭 **token IDs** 就启动 CXL 读 Layer $k$ 的表项

### 和 Step 3 图的关系

[01d 多头寻址图](../diagrams/engram-01d-multi-head-hash.svg) **CPU 区**：$g_{t,n}\to\varphi\to z\to E[\cdot]$ 都只依赖 input → 所以能在 **decode 步入口** 在 CPU 完成，不必等 GPU hidden。

---

## 1. 先纠正两个常见误读

### 误读 A：「prefetch window = CPU 侧缓存」

- **正解**：右侧竖条是 **时间轴**——蓝条 = GPU 在执行前面层；橙条 = 同一时刻 CXL 在拉 Layer $k$ 的数据。
- 条件 $L_{\mathrm{pool}} < \sum_{i=1}^{k-1} t_{\mathrm{exec}}(i)$ 说的是：**池子读延迟** 必须 **小于** GPU 算完 $k-1$ 层的时间，读才能在 Layer $k$ 用到前完成。

### 误读 B：「Path A（CPU memcpy）比 Path B（GPU P2P）更推荐」

[02c 图](../diagrams/engram-02c-cxl-cache-access.svg) Step 2 同时画了：

| 通路 | 做法 | 论文态度 |
|------|------|----------|
| **A** | CXL → **CPU**（OpenMP `memcpy`）→ 再进 GPU | 延迟 **≈ DRAM**，可行 |
| **B** | CXL → **GPU** staging（PCIe **P2P**，`cxl2vram_copy`） | **推荐**：绕过 CPU 搬运、少 launch 开销 |

绿色 **path A** 框 ≠ 「整体以 CPU 为准」；它是 **数据搬运的备选路径**。**点火调度**（Step 0–1）在 CPU/host；**搬运终点**仍是 **L1 HBM staging**。

---

## 2. 为什么 Step 0–1 必须在 CPU/host 侧（Engram 的关键性质）

Engram 索引 **只依赖 input token**（见 [Step 3 图 CPU 区](../diagrams/engram-01d-multi-head-hash.svg)），因此：

1. **Decode 步入口** 就知道 Layer $k$ 要读哪些 `offset[i]`，**不必等** Layer $k-1$ 的 hidden。
2. **CPU 并行算 hash / offsets** 几乎不占 GPU；SGLang 可 **立刻** 向 CXL 发异步读。
3. 若索引像 MoE 路由那样 **依赖 $h_t$**，则必须等前层算完才知道读什么 → **prefetch 无法提前** → 有效窗口 $\approx 0$。

这就是「**CPU 侧先算索引**」对 prefetch **不可或缺**——不是因为 CPU 算力比 GPU 强，而是因为 **这件事本来就不该占 GPU，且可以极早做**（见 [§0 基础概念](#基础概念-decode-layer-hidden)）。

---

## 3. GPU 在 window 里做什么（同样关键）

没有 GPU 在前面层 **实打实算 $t_{\mathrm{exec}}$**，就没有窗口长度：

- Qwen3-32B 例：每步 $\approx 3.6\,\mathrm{ms}$ / 64 层 → **每层 ~56 μs**。
- Engram 若在 **Layer 2**：窗口只有 **Layer 1 那一层** 的时间 ≈ **56 μs**。
- 这 56 μs 内必须完成 CXL 离散读 + 进 staging；故论文强调 **CXL 近 DRAM 延迟**，RDMA 小包不够（图右下角 *why CXL not RDMA*）。→ [完整推导](./cxl-why-cxl-not-rdma.md)

**GPU 越强 / batch 越大 → 前面层越久 → 窗口越宽 → prefetch 越好藏。**

---

## 4. 串起来：一次 decode 步时间线

```
t0  ForwardBatch 到达
    CPU Step0: offsets[] = hash(token_ids)     ← 不占用 GPU
    CPU Step1: SGLang 异步发起 CXL read       ← 目标 gpu_staging

t0..t1  GPU: Layer 0, 1, …, k-1             ← 这就是 prefetch window（蓝条）
        ‖ 并行
        CXL: fetch layer-k embeddings         ← 橙条（Step 2 Path B）

t1  GPU Layer k: staging 已有 e_t
    Step3: W_K,W_V, gate, conv, 写回 H       ← 必须在 HBM
```

**「强」的是整套 overlap**；CPU 负责 **早、准地发起**；GPU 负责 **撑开时间窗 + 消费 staging**。

---

## 5. 和「奠基论文 DRAM offload」一致

单机 Host DRAM offload 时逻辑相同：索引在 CPU 算、prefetch 与前几层 overlap，只是 L3 换成 **L2 DRAM**。CXL 论文把 L3 换成共享池，**window 机制不变**。

---

## 6. 一图对照 02c

| 02c 元素 | 含义 |
|----------|------|
| 绿圈 Step 0–1 | **Host/CPU 编排**（索引 + 异步触发） |
| 黄框 L3 | CXL 池里离散 320B 行 |
| 绿框 path A | 可选：经 CPU memcpy |
| 蓝框 L1 staging | **GPU HBM** 终点 |
| 虚线 P2P preferred | **推荐** 直读 CXL→GPU |
| 右侧 window | **GPU 算 0..k-1 的时间** ‖ **CXL 拉 k** |

---

## 参考

- [engram-series-overview.md §缓存访问逻辑](../engram-series-overview.md#缓存访问逻辑一次-decode-的完整路径)
- [engram-02c-cxl-cache-access.svg](../diagrams/engram-02c-cxl-cache-access.svg)
- [cxl-l1-l2-l3-memory-tiers.md](./cxl-l1-l2-l3-memory-tiers.md)
- CXL 论文 §3.2 时间窗不等式 · arXiv:2603.10087
