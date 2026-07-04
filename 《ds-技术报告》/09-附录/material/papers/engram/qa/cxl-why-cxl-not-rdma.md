# 为何选 CXL 而非 RDMA？

[← 返回 §Step 1 时间窗](../../../../../07-Engram/02-Engram系列导读.md#step-1-异步-prefetch-触发) · [02c 图](../diagrams/engram-02c-cxl-cache-access.svg) · [CXL vs RDMA pattern](cxl-vs-rdma-communication-pattern.md) · [答疑目录](README.md)

---

## 先澄清

**Engram CXL 论文选的是 CXL，不是 RDMA。**

overview 里「这是选 **CXL 而非 RDMA** 的核心原因」= 解释 **为什么不走 RDMA 内存池**（Mooncake 类 KV offload 路径），而要用 **CXL.mem load/store**。

若问「为啥选 RDMA」→ 在 **Engram 大表 + 离散小读 + ~56 μs prefetch 窗** 这个场景下 **不选**；RDMA 仍适合 **KV cache 大块传输** 等别的 workload。

---

## 问题

Step 1 里的时间窗不等式

$$
L_{\mathrm{pool}}(N_{\mathrm{token}}, S_{\mathrm{layer}}) < \sum_{i=1}^{k-1} t_{\mathrm{exec}}(i)
$$

和「Layer 2 仅 ~56 μs」如何推出 **必须 CXL、不能 RDMA**？

---

## 结论

| 约束来源 | 含义 |
|----------|------|
| **时间窗** | Engram 在 Layer $k$，只允许在 GPU 算 Layer $1..k-1$ 的这段时间里，把本层要的 embedding 从 **内存池** 拉进 HBM staging |
| **$L_{\mathrm{pool}}$** | 一次 decode 步、一层 Engram，从池子完成 **所有离散读** 的端到端延迟 |
| **56 μs 例** | Qwen3-32B：$k=2$ 时窗口 $\approx$ 单层时间 $\approx 56\,\mu\mathrm{s}$ |
| **CXL** | $L_{\mathrm{pool}}$ **≈ 本地 DRAM 量级**→ **不等式可满足** |
| **RDMA** | 离散 **320B×多段** + NIC/消息栈 → $L_{\mathrm{pool}}$ **远高于 DRAM** → **56 μs 内完不成** |

**根因两条**：① **延迟预算太紧**（早层 Engram）；② **访问形态是大量小随机读**，不是 RDMA 擅长的大块 get/put。

---

## 1. 时间窗从哪来

一次 **[decode 步](cxl-prefetch-window-cpu-gpu.md#基础概念-decode-layer-hidden)**：

1. CPU 在步入口就算好 `offsets[]`（只依赖 token IDs）。
2. SGLang **异步 prefetch** Layer $k$ 的 Engram 行。
3. GPU **同时**算 Layer $0,1,\ldots,k-1$。
4. 进入 Layer $k$ 时，staging 里 **必须已有** $e_t$。

因此池子读延迟 $L_{\mathrm{pool}}$ 必须小于前面层的 GPU 时间之和：

$$
L_{\mathrm{pool}} < t_{\mathrm{exec}}(1) + \cdots + t_{\mathrm{exec}}(k-1) = \sum_{i=1}^{k-1} t_{\mathrm{exec}}(i)
$$

[02c 图](../diagrams/engram-02c-cxl-cache-access.svg) 右侧蓝条 = 右边求和；橙条 = $L_{\mathrm{pool}}$ 必须落在蓝条长度以内。

---

## 2. Qwen3-32B 的 56 μs 怎么算

| 量 | 值 |
|----|-----|
| 每 decode 步总时间 $t_{\mathrm{step}}$ | $\approx 3.6\,\mathrm{ms}$ |
| 层数 | 64 |
| 单层平均 $t_{\mathrm{exec}}(i)$ | $3.6/64 \approx 56\,\mu\mathrm{s}$ |
| Engram 插在 $k=2$ | 窗口 = $t_{\mathrm{exec}}(1) \approx$ **56 μs** |
| Engram 插在 $k=15$ | 窗口 = $\sum_{i=1}^{14} t_{\mathrm{exec}}(i) \approx 14\times 56 \approx 0.78\,\mathrm{ms}$（宽裕得多） |

**最苛刻的是最早插入层（如 Layer 2）**：几乎只有 **一层** 的 GPU 时间来藏 CXL 读 → 池子延迟必须 **接近 DRAM**，不能是「慢一个数量级的网络内存」。

---

## 3. 为何 RDMA 过不了这个窗

### 3.1 延迟：消息栈 vs load/store

| | **CXL** | **RDMA 池** |
|--|---------|-------------|
| 语义 | `mmap` 后 **load/store** | **get/put** 请求/响应 |
| 软件路径 | 用户态直接寻址 + DMA | bounce buffer、NIC、远端处理 |
| 论文测量 | CXL→CPU **≈ DRAM**；CXL→GPU 略高仍可接受 | 离散小包延迟 **数量级高于 DRAM**（Fig 3） |

在 **56 μs** 预算里，RDMA 一次或多次 round-trip 的固定开销往往就把窗口吃光，更别说 **16 段×320B** 级别的多段读。

### 3.2 粒度：5 KB×离散 vs MB–GB 块

Engram 每层每 token：

- 载荷 $\approx$ **5 KB**（多阶 n-gram × 多头 × 320B/行）
- **稀疏离散** 随机行读

RDMA 池为 **KV cache offload** 优化：**大块、连续** 传输效率高；**64B～数百字节级海量小包** 时：

- 带宽利用率可跌至峰值 **~25% 以下**（overview §② 动机）
- 每次 get/put 有固定成本 → $L_{\mathrm{pool}}$ 随段数恶化

CXL 用 **cache-line 级 load/store**，与「随机抽很多小行」匹配。

### 3.3 带宽不是瓶颈

论文算过：所需池带宽 $\approx T \times S_{\mathrm{layer}} \times N_{\mathrm{eng}} \approx 0.7\,\mathrm{GB/s}$，PCIe Gen5 / 百 G 网卡都够。

**卡的是延迟形态**，不是峰值 GB/s——所以「RDMA 带宽也很大」**不能**替代 CXL。

---

## 4. 为何 CXL 能过

1. **内存语义**：`cxl_base + offset[i]` **[按地址读](cxl-vs-rdma-communication-pattern.md)**，无 per-chunk 消息。
2. **实测延迟**：CXL→CPU ≈ DRAM；CXL→GPU P2P 略高仍 **< 56 μs 级窗口**。
3. **端到端**：Qwen3-4B DRAM **5683.7** vs CXL **5614.4** tok/s（**~1.2%** 损），说明换 CXL 池 **几乎不掉吞吐**。
4. **多机**：Switch 映射共享 L3，仍保持 load/store 延迟曲线，而非 RDMA 小包曲线。

---

## 5. RDMA 适合什么

| Workload | 更合适的技术 |
|----------|----------------|
| **KV cache** 大块 offload / 跨机共享 | RDMA 池（Mooncake 等） |
| **Engram embedding 表** 离散小读 + 紧 prefetch 窗 | **CXL.mem** |
| 单机 Host DRAM 表 | 奠基论文：DRAM + PCIe prefetch（无 CXL 硬件时） |

同一推理栈里可以 **KV 用 RDMA、Engram 用 CXL**——争的是 **不同访问形态**，不是二选一谁更强。

---

## 6. 决策链

```
Engram 索引只依赖 input ids
 -> decode 步入口即可 prefetch
 -> 有效窗口 = sum t_exec(1..k-1) [有时仅 ~56 μs]
 -> 需要 L_pool ≈ DRAM
 -> CXL load/store 满足；RDMA get/put 不满足
 -> 选 CXL 池（Fig 2b），不选 RDMA 池（Fig 2a）
```

---

## 参考

- [DeepSeek Engram 系列导读§Step 1](../../../../../07-Engram/02-Engram系列导读.md#step-1-异步-prefetch-触发)
- [DeepSeek Engram 系列导读§与 RDMA 对比](../../../../../07-Engram/02-Engram系列导读.md#与-rdma-池的访问逻辑对比)
- [CXL vs RDMA：Engram 的两种「远程内存」通信 pattern](cxl-vs-rdma-communication-pattern.md)
- [Prefetch window：不是「CPU 比 GPU 强」，而是 CPU 先点火、GPU 腾出时间窗](cxl-prefetch-window-cpu-gpu.md)
- CXL 论文 §3.2 时间窗、Figure 2a/2b/3 · arXiv:2603.10087
