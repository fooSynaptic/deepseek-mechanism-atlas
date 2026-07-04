# Lightning Indexer 详解

> [← 中文导读](../00-前言/02-中文导读.md) · [← 仓库首页（EN）](../../README.md) · [← DSA 系列导读](01-系列导读.md) · [← 演进总览 §3.6](../01-总览/01-版本演进总览.md#36-deepseek-v32--v32-exp)

> [← DSA 梗概](02-DSA梗概.md) · [DSA 逻辑](03-DSA逻辑详解.md) · [§1 在 DSA 中的角色](#1-在-dsa-中的角色) · [§1 答疑：常数极小](#答疑为何-oleftl2right-却说常数极小) · [Decode 一步 walkthrough](#decode-forward-walkthrough)  
> **论文**：[DeepSeek-V3.2 arXiv:2512.02556](https://arxiv.org/pdf/2512.02556) · **第三方**：[Raschka §4](../08-外部解读/02-Raschka全文解析.md#4-deepseek-v32-exp-and-sparse-attention)

---

## 1. 在 DSA 中的角色

**Lightning Indexer** 是 [DeepSeek Sparse Attention (DSA)](02-DSA梗概.md) 的**第一阶段**：在 **Core MLA** 做昂贵、精度敏感的主 attention 之前，用**极低成本**决定「当前 token 该 attend 哪些历史位置」，供 **Top-$k$ Selector** 选出 $k{=}2048$ 个历史下标。

每层 DSA 流水线：**Indexer 选位置** → Top-$k$ 得 index 集 $I$ → **Core MLA 只对 $I$ 内 entry 算 attention**。Indexer 与 Core MLA 读的是**两套 cache**（Indexer-Cache vs Latent-Cache），分工见 [§4 Indexer-Cache](#4-indexer-cache存什么为何不-offload) 与 [ess-dual-cache 图](#decode-一步分工)。

### 谁对谁打分？

Decode 推**当前** token $t$ 时，Indexer 做的是：**一个**当前 query，对**全长历史**里**每一个**历史位置 $s$ 各打一个分。

| 侧 | 符号 | 从哪来 | 角色 |
|----|------|--------|------|
| **Query 侧（当前）** | $q_t$（或 $q_{t,j}$，$j$=indexer 头） | 本步 $h_t$ **现算**，不进 cache | **打分方**：「我现在要 attend 谁？」 |
| **Key 侧（历史）** | $k_s$ | **Indexer-Cache 第 $s$ 行**（历史 token $s$ 预存） | **被打分方**：「历史第 $s$ 个 token 有多相关？」 |
| **输出** | $I_{t,s}$ | 对每个 $s=1\ldots L$ 一个标量 | **当前 $t$** 认为 **历史 $s$** 应进入 Top-$k$ 的分数 |

一句话：**$q_t$ 对每条历史的 $k_s$ 打分** → 得到 $L$ 个 $I_{t,s}$ → Top-$k$ 得 $I$ → Core MLA 只对 $I$ 内 Latent-Cache 做 attention。

> **方向**：**$q_t \to k_s$**（当前 query 选历史 key），**不是** $k_s$ 选 $q_t$，也**不是**每个历史 token 各自做 top-$k$。

| 对比 | Lightning Indexer | Core MLA |
|------|-------------------|----------|
| 目的 | **决定看谁** | **算 attention 输出** |
| 扫过的历史 | **全长** $L$ 个历史 token | **仅** $I$ 内 $k$ 个历史 token |
| 典型复杂度 | $O(L^2)$ 量级，**常数极小** | $O(Lk)$，$k$ 固定 |
| 是否 softmax | **否** | **是**（精度主路径） |
| 是否读 512 维 $c^{KV}$ | **否**（读 indexer 专用 cache） | **是**（升维后做 MLA） |

### 答疑：为何 $O(L^2)$ 却说「常数极小」？

表里写 Indexer **典型复杂度 $O(L^2)$ 量级、常数极小**，容易读成矛盾：既然还是二次，凭什么说「极轻」？  
要点：$O(\cdot)$ 只描述随 $L$ 增长的**阶**；真正决定耗时的是前面的**系数**（每对 $(t,s)$ 做多少算、读多少字节、走多少条算子）。Indexer 与稠密 MLA **同阶** $L^2$，但系数通常小 **一个数量级以上**。

#### 1. 「$O(L^2)$」在这里指什么？

| 模式 | Indexer 在做什么 | 量级 |
|------|------------------|------|
| **Prefill** | 序列里**每个**位置 $t$ 都要对全部可见历史 $s$ 打分 | **$O(L^2)$**（主瓶颈常在这里） |
| **Decode 一步** | **一个**当前 $q_t$ 对全长 $L$ 个 $k_s$ 打分 | **$O(L)$** per step |

因此「$O(L^2)$」主要是 **Prefill / 训练**口径；Decode 每步是线性的，但**每层仍要扫全长 Indexer-Cache**。表里与 [dsa-logic](03-DSA逻辑详解.md) 沿用同一说法：**相对全长历史做 routing，阶与稠密 QK 相同，常数不同**。

#### 2. 常数小在哪？（相对 Core MLA / 稠密 MLA）

论文 ([2512.02556](https://arxiv.org/pdf/2512.02556) §DSA) 原话：indexer 仍是 $O(L^2)$，但 **requires much less computation compared with MLA**；且 **small number of heads、可 FP8 实现**。本地实现与图（[§2 walkthrough](#decode-forward-walkthrough)）可拆成下面几条：

| 维度 | Lightning Indexer | Core MLA（稠密对照） |
|------|-------------------|----------------------|
| **每 token 读什么** | Indexer-Cache 里 **低维** $k_s$（[§4](#4-indexer-cache存什么为何不-offload) 约占 KV **~16.8%**） | Latent-Cache **512-d** $c^{KV}$ + RoPE，再 **升维** 到 128 头 |
| **头数** | $H^I$ **很少**（远小于 MLA **128** 头） | 128 query 头完整 attention |
| **单步算子** | $\sum_j w_{t,j}\,\mathrm{ReLU}(q_{t,j}\cdot k_s)$：**点积 + ReLU + 加权求和** | QK$^\top$ → **softmax** → 乘 **$V$**（精度主路径） |
| **是否 softmax / $V$** | **否** | **是** |
| **主 attention 扫多长** | 必须 **全长** $L$（为选 Top-$k$） | DSA 下 **仅** $k{=}2048$ 条（$O(Lk)$） |

一句话：**Indexer 是「廉价、低维、无 softmax 的全长打分」；贵的是后面只对 $k$ 条做的 MLA。** 系数小来自 **更窄的向量、更少的头、更少的算子**，而不是 $L$ 的阶更低。

#### 3. 粗算对比（量级直觉）

记每个 $(t,s)$ 对的 **Indexer** 代价 $\propto H^I \cdot d_I$（$H^I$ 个头、每头 $d_I$ 维点积；V3.2 官方实现里 $H^I$ 小、$d_I$ 低，且可走 **FP8**）。  
**稠密 MLA** 的 QK 一项 $\propto H^{\mathrm{MLA}} \cdot d_{\mathrm{head}}$，且之后还有 **softmax + 对 $V$ 的加权**（$V$ 维与 latent 升维同量级）。

再乘 **cache 带宽**：Indexer 只读 **Indexer-Cache** 行；MLA 读 **512-d latent** 并展开。ESS 论文给出异构占比 **Indexer ~16.8% / Latent ~83.2%**，单 token 的 indexer  footprint 约为 latent 的 **~1/5**，Prefill 全长扫 cache 时 **内存流量**也差一截。

所以：**阶仍是 $L^2$，但「每格」便宜一个数量级以上** —— 这就是「常数极小」的含义。

#### 4. 既然这么轻，为什么 Index Share 还要优化它？

常数小 **≠** 可忽略：

- **Prefill**：层数 $\times$ 全长 $\times$ 每层独立 indexer → 长上下文下 indexer 可占 **Prefill 大部分时间**（IndexCache 论文引 V3.2 场景约 **~81%**；见 [§8](#8-推理与-infra)）。
- **Decode**：每步 $O(L)$ 且 **每层都跑**；$L{=}128\text{K}$ 时累加仍可观。
- **DSA 总收益**：Indexer 贵但 **只 routing**；Core MLA 从 $O(L^2)$ 收到 **$O(Lk)$**（$k{=}2048$），端到端长上下文仍明显变快（论文 Figure 3 服务 benchmark）。

[Index Share](06-Index-Share逻辑.md) 动的是 **重复 indexer 算子**（层间复用 Top-$k$），不是否认「单算 indexer 很轻」，而是 **F 层算一次、S 层跳过**。

#### 5. 和推荐 ETA 的同一句话

两阶段都是 **cheap score over full history → expensive attention on top-$k$**（与推荐场景 **ETA** 同构）。ETA 里 Stage-1 也常常是 **低维内积 / 双塔**，Stage-2 才是 **完整 target attention**；DSA 的「常数极小」就是 Stage-1 故意做 **极简打分器**。

---

<a id="decode-forward-walkthrough"></a>

## 2. Decode 一步 walkthrough：$t$、$s$ 怎么完成？

上面是概念层：**固定当前 $t$，遍历历史 $s$ 打分，Top-$k$ 选下标**。下面用 **$L=8$、当前 $t=9$** 的一次 decode 前向（实际 $L$ 可达 128K），把**输入 → 前向 → 输出**三块与实现对齐；读图时从左到右对应下文的 §2.1–§2.3。

<img src="figures/lightning-indexer-forward.svg" alt="Decode 一步 Indexer 前向：固定 t=9，遍历 s=1..L 得 I_{t,s}，Top-k 得 I；输入 h_t 与 Indexer-Cache，输出分数向量与 index 集" width="920"/>

[直接打开 SVG](figures/lightning-indexer-forward.svg)

### 2.1 输入是什么？（图左栏）

| 输入 | 形状 / 含义 | 何时产生 |
|------|-------------|----------|
| $h_t$（例 $h_9$） | [5120] 当前隐状态 | **本步**由上一层输出 |
| Indexer-Cache 第 $s$ 行 $k_s$ | [d_I] indexer key | **历史**：token $s$ 成为当前步时已写入 |
| Latent-Cache $c_s^{KV}$ | [512] + $k^R$ | Indexer **不读**；Core MLA 升维用 |

历史 $s=1\ldots 8$ 在各自成为「当前步」时已完成 **K 写入**（与 MLA 主路径并行、更轻）：

$$
k_s = W^{IK}\,\mathrm{proj}(h_s) \quad \Longrightarrow \quad \text{append Indexer-Cache 第 } s \text{ 行}
$$

本步 $t=9$ **不重算** $k_1\ldots k_8$，只读 cache。

### 2.2 前向怎么完成 $t$、$s$ 操作？（图中栏）

**固定当前 $t=9$**，对 **每个历史** $s \in \{1,\ldots,L\}$ 各算一个分：

$$
I_{9,s} = \sum_{j=1}^{H^I} w_{9,j}\,\mathrm{ReLU}(q_{9,j} \cdot k_s)
$$

| 步骤 | 操作 | $t$ / $s$ 含义 |
|------|------|----------------|
| **A** | $q_9 = W^{IQ}\,\mathrm{slice}(h_9)$ | **$t$ 只出现一次**：现算当前 query |
| **B** | $s=1,2,\ldots,L$（实现为 batched GEMM） | **$s$ 在循环里扫**：同一 $q_9$ 点积 L 个 $k_s$ |
| **C** | Top-$k$（$k{=}2048$） | 取 $I_{9,s}$ 最高的 $k$ 个**历史下标** $s$，得 $I$ |

- **DeepGEMM**：`dot(q_t, Indexer-Cache)` 一次产出长度 $L$ 的分数向量
- Indexer **不算** softmax / $V$ / 512 维 $c^{KV}$（留给 Core MLA）

### 2.3 输出是什么？（图右栏）

| 输出 | 形状 | 下游 |
|------|------|------|
| 分数向量 $[I_{t,1},\ldots,I_{t,L}]$ | $[L]$ | 中间量；Top-$k$ 后立即丢弃大部分 |
| **index 集 $I$** | $\|I\|=k$ 个 $s$ | **Core MLA** 只读 Latent-Cache 中 $s \in I$ 的行 |
| （本步 append）$k_t$、$c_t^{KV}$ | 各 1 行 | 供下一步 $t{+}1$ 当历史 $s$ 使用 |

$L=8$ 时 Top-$k$ 最多 $\min(2048,L)=8$；$L=32\text{K}$ 时 $k{=}2048$ 才有稀疏收益。

### 2.4 Prefill 与 Decode

| 模式 | 当前 $t$ | 可见历史 $s$ | Indexer |
|------|----------|--------------|---------|
| **Prefill** 第 $t$ 行 | batch 内位置 $t$ | $1 \le s < t$ | 对前缀打分 → Top-$k$ → Core MLA |
| **Decode** 一步 | 新 token，$t=L{+}1$ | $1 \le s \le L$ | 对全长 Indexer-Cache 打分 → Top-$k$ |

每处理完 token $t$，**append** $k_t$ 到 Indexer-Cache（与写 $c_t^{KV}$ 到 Latent-Cache 同步发生，Indexer 路径更轻）。

<a id="decode-一步分工"></a>

<img src="figures/ess-dual-cache.svg" alt="MLA Decode 一步: Indexer 选 top-2048 位置 vs Latent-Cache 升维并做 Core MLA" width="920"/>

[直接打开 SVG](figures/ess-dual-cache.svg)

> **左栏**：Indexer 路径（与上文 walkthrough 一致）；**右栏**：Latent-Cache 升维 + Core MLA，只吃 $I$ 内 entry。

---

## 3. 「位置」、$t$ / $s$ 与 Top-$k$ 方向

### 「位置」= 序列里第几个 token 的下标

不是空间坐标，也不是 attention head 编号。  
已生成序列长度 $L$ 时，每个 token 占序列中的一个**槽位**；槽位编号就是下标（walkthrough 用 $1\ldots L$，公式里常写 $0 \le s < t$）。

| 符号 | 指谁 | 在 decode 一步里 |
|------|------|------------------|
| **$t$** | **当前**正在算的 token | 本步新生成的那个 query（来自 $h_t$） |
| **$s$** | **历史**序列里的某个 token | Indexer-Cache / Latent-Cache 里第 $s$ **行**（过去某 token 存下的向量） |
| **$j$**（公式 $q_{t,j}$） | **Indexer 的第 $j$ 个头** | 与历史下标 $s$ **无关**；仅多头打分时出现 |

图中历史下标有时写作 $j$（如 `indexer_j`），与公式里的 **$s$ 同义**；不要与 $q_{t,j}$ 里的头编号 $j$ 混用。

### Top-$k$：谁对谁选？

**一句话**：**当前 step 的 query，在全部历史 token 里选出 $k$ 个最相关的下标**（与 [§2 walkthrough](#decode-forward-walkthrough) 一致）。

Decode 每推一个新 token，$t$ 前进一格，**重新**对更新后的全长历史做一遍 Top-$k$（[Index Share](05-Index-Share梗概.md) 可在层间复用 $I$，那是 infra 优化，不改变「$q_t$ 选历史 $s$」的方向）。

---

## 4. 打分公式（论文 / Raschka 整理）

对当前 query 下标 $t$、历史 token 下标 $s$（$0 \le s < t$），重要性分数：

$$
I_{t,s} = \sum_{j=1}^{H^I} w_{t,j}\,\mathrm{ReLU}\!\left(q_{t,j} \cdot k_s\right)
$$

| 符号 | 含义 |
|------|------|
| $H^I$ | **Indexer 头数**（远小于 MLA 的 128 头；低维、廉价） |
| $q_{t,j}$ | **当前 token** $t$ 上、第 $j$ 个 **indexer head** 的 query（$j$=头编号） |
| $k_s$ | **历史 token** $s$ 的 indexer key（Indexer-Cache **第 $s$ 行**；非 MLA 主 KV） |
| $w_{t,j}$ | 可学习的 per-head 标量权重 |
| $I_{t,s}$ | **当前 $t$** 认为 **历史 $s$** 有多重要（分数越高越应进入 Top-$k$） |

**实现要点**（[Raschka 全文 §4](../08-外部解读/02-Raschka全文解析.md#4-deepseek-v32-exp-and-sparse-attention)）：

- Indexer **只对 query 侧多头**；keys 已在 cache，按历史下标 $s$ 查第 $s$ 行即可。
- **ReLU** 不保证分数为 0；**真正稀疏来自 Top-$k$**，不是 ReLU 截断。
- Top-$k$ 的 $k$ **与 GQA/MQA 的 KV 头数无关**；V3.2 官方实现 **$k{=}2048$**。

---

## 5. Indexer-Cache：存什么、为何不 offload

DSA 把 KV 拆成两类（[dsa-logic.md §4](03-DSA逻辑详解.md#4-异构-kv-cacheindexer-cache-与-latent-cache)）：

| | **Indexer-Cache** | **Latent-Cache** |
|--|-------------------|------------------|
| 内容 | 每条历史 token 的 **indexer 轻量向量**（打分用） | MLA **$c^{KV}$** [512] + $k^R$ [64] |
| 占比（ESS 论文） | ~**16.8%** | ~**83.2%** |
| 每步访问 | **全长** $L$ 条（indexer 必须全扫） | **仅** $I$ 内 $k$ 条 |
| GPU 常驻 | **是** | 可 [ESS offload](../06-推理基础设施/01-ESS概念.md) |

Lightning Indexer **只读 Indexer-Cache**，不升维、不读 512 维 latent，因此单步点积带宽远低于「对全长做 MLA」。

---

## 6. 与滑动窗口 / 稠密 MLA 的区别

| | 滑动窗口 | 稠密 MLA（V3.1-T） | DSA + Lightning Indexer |
|--|----------|-------------------|-------------------------|
| 可见历史 | **固定局部**窗 | **全长** | **学到的** $k$ 个位置 |
| 选择规则 | 距离 | 无（全 attend） | indexer 分数 + top-$k$ |
| 能否非局部 | 否 | 是 | **是**（内容相关） |
| 典型 $k$ / 窗宽 | 窗宽 $w$ | $L$ | **$k{=}2048$** |

DSA 的 $k$ 个位置是 **数据驱动、内容相关** 的，不是固定局部窗口（[Raschka 表 4-1](../08-外部解读/02-Raschka全文解析.md#表-4-1滑动窗口注意力-vs-dsa)）。

---

## 7. 训练：如何不损精度

V3.2-Exp 在 V3.1-Terminus 上 **续训加入 DSA**（非从零训练）。Indexer 需学会近似稠密 MLA 的「看谁」模式：

- 论文叙述：用 **KL 散度** 等目标，让 indexer 选出的分布对齐稠密 attention 模式（详见 [2512.02556](https://arxiv.org/pdf/2512.02556) §DSA）。
- 结果：公开 benchmark 与 V3.1-Terminus **基本持平**，长上下文 **算力显著下降**（[V3.2 梗概](../04-版本代际/02-V3.2-DSA.md)）。

---

## 8. 推理与 infra

| 组件 | 与 Lightning Indexer 的关系 |
|------|----------------------------|
| **[DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)** | indexer **logit / 点积** kernel（含 paged 版） |
| **Indexer-Cache 常驻 GPU** | 每 decode step 全长扫；**不可**像 Latent 那样 offload |
| **[Index Share](05-Index-Share梗概.md)** | 跨层 **复用 top-$k$ index**，减 **重复跑 indexer 算子** |
| **Prefill 瓶颈** | 长上下文下 indexer 可占 Prefill **~81%**（IndexCache 论文引 V3.2 场景） |

Indexer **不做**：softmax、latent 升维、读 $c^{KV}$、offload（见 [ess-dual-cache 左栏](#decode-一步分工)）。

---

## 9. 上下游

| 方向 | 文档 |
|------|------|
| 前置 | [MLA](../02-基座架构/02-MLA低秩注意力.md) · [V3.1 Hybrid](../04-版本代际/01-V3.1-Terminus.md) |
| 同流水线 | [Top-$k$ + Core MLA](03-DSA逻辑详解.md#32-top-k-selector) |
| 下游 infra | [Index Share](06-Index-Share逻辑.md) · [ESS](../06-推理基础设施/01-ESS概念.md) |
| 下游算法 | [CSA/HCA 专文 §2](../04-版本代际/05-CSA-HCA混合压缩注意力.md#csa-compressed-sparse) · [V4 梗概](../04-版本代际/03-V4.md)（新一代压缩 + 稀疏） |

---

## 10. 参考

1. DeepSeek-AI. *DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models.* arXiv:2512.02556, 2025.
2. Sebastian Raschka. *From DeepSeek V3 to V3.2* — [§4 Sparse Attention](../08-外部解读/02-Raschka全文解析.md#4-deepseek-v32-exp-and-sparse-attention) · [要点速读](../08-外部解读/01-Raschka要点速读.md#与本地文档映射)
3. 本地图：见 [§2 walkthrough 图](#decode-forward-walkthrough) · [§2 ess-dual-cache 图](#decode-一步分工) · [dsa-pipeline（DSA 逻辑）](03-DSA逻辑详解.md)

---

## 章节导航

| ← 上一章 | 下一章 → |
|----------|----------|
| [DSA（DeepSeek Sparse Attention）逻辑详解](03-DSA逻辑详解.md) | [Index Share（IndexCache）梗概](05-Index-Share梗概.md) |

> [← 中文导读](../00-前言/02-中文导读.md) · [← 仓库首页（EN）](../../README.md) · [← DSA 系列导读](01-系列导读.md) · [← 演进总览 §3.6](../01-总览/01-版本演进总览.md#36-deepseek-v32--v32-exp)
