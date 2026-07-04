# MoE 推理：Expert Parallel（EP）与 gather/scatter

[← 返回 Hash MoE §1.3](../hash-moe-fp4.md#still-inherited) · [DeepSeekMoE §推理 infra](../deepseek-moe.md#推理-infra-关注点) · [答疑目录](./README.md)

---

## 一句话

**Expert Parallel（EP）** 把 **routed experts 的权重** 切到多张 GPU 上；前向时 token 经 **dispatch（scatter）** 送到持有对应 expert 的 rank，算完再 **combine（gather）** 回 batch 布局。DeepSeek V3 / V4 Hash MoE **共享这条 MoE 推理骨架**——变的是 **怎么选 expert id**（centroid vs hash），不是 EP 通信模式本身。

---

## 1. EP 是什么

| 并行维 | 切什么 | MoE 里典型用途 |
|--------|--------|----------------|
| **Tensor Parallel（TP）** | 单层内矩阵 **列切 + 行切**（见下） | Attention / 大线性层 |
| **Expert Parallel（EP）** | **不同 expert 的 FFN 权重** 分到不同 GPU | routed MoE 层 |
| **Data Parallel（DP）** | 复制整模型、切 batch | 通用训练 |

> **TP 补注（列/行不是二选一）**  
> 表里「按列/行切」指 **Megatron 式 TP 在同一层里通常各用一次**，不是「整张矩阵只切一种方向就结束」：
>
> - **列切（column parallel）**：权重按 **输出维 / 列** 分片；每卡算 $Y_i = X W_i$，得到 **输出通道分片**（常接 AllGather 或留分片给下一步）。
> - **行切（row parallel）**：权重按 **输入维 / 行** 分片；每卡算 **部分和**，需 **AllReduce** 合并成完整输出。
>
> 典型 FFN / 投影块：**先列切、后行切**（或 Attention 里 QKV 列切 + $O$ 投影行切），**至少两次 TP matmul + 中间通信** 才走完一层，且 **列切那一次与行切那一次分工不同**——不是对同一矩阵重复切两次。

DeepSeekMoE 每层有 **256 routed + shared**（V3 口径）：**shared** 通常各 rank 都有或按 TP 复用；**routed** 池太大，单卡放不下 → 用 **EP** 按 expert id 分片。

---

## 2. 前向数据流：gather / scatter

对单个 MoE 层、单个 micro-batch，顺序为：**路由得 expert id** → **dispatch（scatter）** 把 token 发到持有该 expert 的 GPU → **各 rank 算本 rank expert 子集** → **combine（gather）** 加权收回原 batch 顺序。

**Shared experts** 不走稀疏 scatter：每层 **恒激活**，与 routed 输出 **相加**（见 [DeepSeekMoE](../deepseek-moe.md)）。

---

## 3. 为何与负载均衡绑在一起

EP 下 **每张卡只服务一部分 expert**。若路由 collapse（多数 token 涌向少数 expert）：

- 热点 rank **打满**，其它 rank **空转** → **EP 效率 ≈ 最忙那张卡**。
- 通信量也随 **激活 expert 的 rank 分布** 波动。

因此 V3 的 [aux-loss-free](../aux-loss-free-moe-routing.md) / [$L_{\mathrm{Bal}}$](../moe-sequence-wise-balance-loss.md) 首要目标之一是 **batch 级 expert 均衡**——直接服务 **EP 训练与推理吞吐**。

---

<a id="4-hash-moe-改什么不改什么"></a>

## 4. Hash MoE 改什么、不改什么

| 维度 | V3 centroid 路由 | V4 Hash 层 |
|------|------------------|------------|
| **Expert id 怎么来** | $u^\top e_i$ + top-$K_r$ | **确定性 hash**（无 router GEMM） |
| **EP scatter/gather** | ✅ 同族 | ✅ **仍走** routed gather-scatter |
| **Shared 双路径** | ✅ | ✅ |
| **负载均衡手段** | $b_i$、$L_{\mathrm{Bal}}$ | hash 函数 **静态近似均匀** + 深层仍可 centroid |

> **读法**：[Hash MoE 专文](../hash-moe-fp4.md) 换的是 **浅层选 expert 的函数**；**一旦 id 确定**，后面仍是 DeepSeekMoE 的 **EP + shared** 引擎路径，与 V3 推理栈同族。

> **答疑**：[为何只改浅层、深层仍 centroid？](./hash-moe-shallow-vs-deep.md)

---

## 5. V4 叠加：FP4 + EP

[Hash MoE + FP4](../hash-moe-fp4.md#fp4-moe-quant) 在 EP 路径上多一层约束：

- **Routed expert 权重** FP4 存 HBM；各 rank 的 expert 子集 **4bit 读 + 高精度累加**（引擎需 **分 kernel**）。
- **Shared** 路径通常仍用更高精度（与 [V3 FP8](../v3-fp8-dynamic-quantization.md) 栈衔接）。
- EP **通信 payload** 仍是 **activation**（hidden），不是 expert 权重；FP4 主要省 **权重带宽与显存**，不改变 scatter/gather 拓扑。

---

## 6. 与训练侧 DeepEP 的关系

V3 训练文档常并列 **DualPipe、DeepEP、FP8**（见 [v3.md §排除项](../v3.md#v3-structure-excluded)）：**DeepEP** 是 **MoE token dispatch 的通信库**，属于 **训推系统层**，不是 Transformer 权重结构。

| 层 | 内容 |
|----|------|
| **模型结构** | 256/8 routed、shared、路由公式 |
| **EP / DeepEP** | expert 分片 + token **怎么跨 rank 搬** |
| **Hash MoE** | 仅改 **部分层** 的 id 映射函数 |

---

## 7. 对照表

| 术语 | 含义 |
|------|------|
| **EP** | Expert Parallel；按 expert 切权重 |
| **scatter / dispatch** | token → 各 expert rank |
| **gather / combine** | expert 输出 → 回原 batch |
| **DeepEP** | DeepSeek 系 MoE 通信实现（训练/推理栈） |
| **device-limited routing** | V2 限制 token 只打本地 expert，减 EP 通信（见 [fine-grained 答疑](./moe-fine-grained-segmentation.md)） |

---

## 参考

- [DeepSeekMoE](../deepseek-moe.md) · [aux-loss-free](../aux-loss-free-moe-routing.md)
- [Hash MoE + FP4](../hash-moe-fp4.md)
- Megatron MoE：[FlexDispatcher + DeepEP](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/README.md)
