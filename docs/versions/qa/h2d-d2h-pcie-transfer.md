# H2D / D2H 是什么？

[← 返回 ESS 论文梗概 §Figure 6&7](../ess-paper-highlights.md#figure-67-overlap-策略对比) · [← ESS 梗概 §Fig.3 / PD 时序](../ess-paper-highlights.md#系统划分fig3) · [答疑目录](./README.md)

---

## 1. 定义

| 缩写 | 英文 | 方向 | 典型含义 |
|------|------|------|----------|
| **H2D** | **Host-to-Device** | CPU 主机内存 → GPU 显存 | 把数据 **搬到 GPU** 上算 |
| **D2H** | **Device-to-Host** | GPU 显存 → CPU 主机内存 | 把结果或 cache **写回 CPU** |

在 CUDA 语境里，二者常通过 **PCIe**（ESS 论文为 **PCIe 5.0**）或 **NVLink** 完成；API 层常见 `cudaMemcpyAsync(..., cudaMemcpyHostToDevice)`（H2D）与 `cudaMemcpyDeviceToHost`（D2H）。

---

## 2. 在 ESS / V3.2 里指什么

ESS 把 **Latent-Cache** 大头放在 **CPU DRAM**，GPU 只留 **Sparse Memory Pool**（LRU 热子集）。每个 decode step：

1. **Indexer**（GPU）选出本步需要的 top-$K$ latent **index**；
2. **H2D（prefetch）**：Cache **miss** 的 latent entry（约 **656B/条**）从 CPU **拉进** GPU 热池；
3. **Core MLA Attention** 在 GPU 上算；
4. **D2H（写回）**：本步新生成的 latent **写回** CPU 侧 Total Memory Pool。

因此 Fig.3 时间线上的 **H2D prefetch**、**D2H 写回** 都是 **Latent-Cache 在 CPU↔GPU 间的搬运**；**Indexer-Cache 不 offload**，也不走这条搬运链。

---

## 3. 为何 ESS 论文反复强调 H2D

| 问题 | 说明 |
|------|------|
| **小块、分散** | 每步多达 $K{=}2048$ 次、每次仅 **656B** → 原生 `cudaMemcpyAsync` H2D 有效带宽极低（论文约 **0.79 GB/s**） |
| **与算力串行** | SGLang 默认：**Indexer → H2D → Attention** 全串行 → GPU 在等传输时空转 |
| **优化手段** | **FlashTrans + UVA** 拉高 H2D/D2H 带宽；**DA/DBA Overlap** 让 **Attention 与 H2D 并行**（如 Attn0 $\parallel$ H2D） |

论文报告优化后 H2D 约 **0.79 → 37 GB/s**，D2H 约 **0.23 → 43 GB/s**（Table/§3 工程数据，见 [梗概 §三大工程模块](../ess-paper-highlights.md#三大工程模块对应-3)）。

---

## 4. 和「Overlap」的关系（你截图里的那句）

**无 Overlap（默认）**

```text
Indexer 完成 ──► H2D 完成 ──► Attention 开始
         （全串行，GPU 常在等 H2D）
```

**DA Overlap**

- 把 Attention 拆成 **Attn0**（只用 GPU 里已有的 latent）和 **Attn1**（等 prefetch 完成）；
- **Attn0 与 H2D 同时进行** → 用计算掩盖一部分传输延迟。

**DBA** 再在 batch 维切 Indexer，长上下文下算力更足，进一步掩盖 H2D。

---

## 5. 相关概念（对照）

| 概念 | 与 H2D/D2H 关系 |
|------|----------------|
| **Prefetch** | 在需要算 Attention 之前 **提前发起 H2D**，把 miss 的 latent 拉上来 |
| **UVA（Unified Virtual Addressing）** | GPU 内核直接访问 **pinned** 的 CPU 地址，减少小块 Memcpy 开销 |
| **PCIe 带宽** | H2D/D2H 的物理上限；ESS 针对「大量小传输」做软件栈优化 |

---

## 6. 延伸阅读

- [ess-paper-highlights.md](../ess-paper-highlights.md) — Fig.3 时序、Fig.6–7 Overlap、FlashTrans 数据
- [ess-latent-cache-offload.md](../ess-latent-cache-offload.md) — ESS 概念与双 Cache 分工

**论文**：[arXiv:2512.10576](https://arxiv.org/abs/2512.10576)
