# FP8 动态量化里的 partial sum 漂移

[← 返回 FP8 专文 §为何需要动态量化](../v3-fp8-dynamic-quantization.md#为何需要动态量化) · [答疑目录](./README.md)

---

## 1. partial sum 是什么

矩阵乘 $Y = X W$ 里，**每一个输出标量**都是一条内积：

$$
Y_{i,j} = \sum_{k=1}^{K} X_{i,k}\, W_{k,j}
$$

硬件做 GEMM 时不会一次性算完整个 $K$，而是按 **[MMA（Matrix Multiply-Accumulate）](./fp8-mma-term.md)** 分块：每来一小块 $(X_{i,k}, W_{k,j})$ 就 **乘加进同一个累加寄存器**。这个 **尚未算完的中间和**，就是 **partial sum**（部分和）。

DeepSeek-V3 在 **FP8 Tensor Core** 上跑这些 MMA；Figure 7(b) 的 WGMMA 1–4 就是在同一条输出维度上 **连续收 partial sum**。

---

## 2. 「漂移」指什么

**漂移（drift）** = partial sum 相对 **理想 FP32/BF16 内积** 的偏差，随着 MMA 次数变多 **单调变大**，最后写回的 $Y_{i,j}$ 系统性偏离真值。

根因不是「FP8 乘一次就不准」这么简单，而是 **低精度累加链太长**：

| 环节 | 发生了什么 |
|------|------------|
| 每次 MMA | FP8（或 TC 内更低精 acc）乘加 → **舍入一次** |
| 重复 $K$ 次 | 误差 **累积**，不是随机抵消 |
| $K$ 很大 | V3 的 hidden / expert 宽度 $\gg 128$，一条内积要 **成千上万次** MMA |

直觉：像用 **短尺子** 一段段量长跑，每段都有四舍五入，段数越多，终点离真实长度越远。

---

<a id="ab-implementation"></a>

## 3. 和 (a) 细粒度 scale 的分工

专文 Figure 7 两路逻辑 **解决不同环节的问题**：

| 机制 | 作用阶段 | 针对误差 | V3 实现要点 |
|------|----------|----------|-------------|
| **(a) Fine-grained scale** | **乘之前** 量化 | 激活离群导致 FP8 **表示**失真 | 激活/权重按 **$N_c{=}128$** 切块；每块算 **dynamic scale** $s_x, s_w$（由块内 absmax 定标）；块内 round 到 **FP8** 后送 **Tensor Core MMA**；**CUDA Core** 对 MMA 输出乘 **$s_x \cdot s_w$** 反量化 |
| **(b) FP32 promotion** | **乘之后** 累加 | MMA 链上 **partial sum 漂移** | **WGMMA** 在 TC 内用 **低精度 acc** 沿 $K$ 维收 partial sum；**每 128 个 MMA 元素** flush 到 **CUDA Core FP32 寄存器** 续加；TC acc **清零** 再收下一段；整条内积 = 多段 FP32 部分和之和 |

(a) 保证送进 MMA 的 FP8 块「尽量满量程」；(b) 保证 **加起来的过程** 不会因为 TC 低精度 acc 太长而漂。

**实现上两路在同一次 GEMM 内核里串联**：先 (a) 按块量化并发起 MMA，MMA 流水里嵌入 (b) 的 **128 步 promotion**；反量化与 FP32 累加都在 **CUDA Core** 完成，Tensor Core 只负责 FP8 乘加吞吐（见 [FP8 专文 §Figure 7](../v3-fp8-dynamic-quantization.md)）。

二者 **都要**；只做 (a) 仍可能在宽矩阵上累加崩，只做 (b) 仍可能被 outlier 块量化拖垮。

---

## 4. 数学上误差怎么叠（简化）

设第 $t$ 步 MMA 的真实贡献为 $p_t$，低精度累加器得到 $\hat{s}_t$：

$$
\hat{s}_t = \mathrm{round}\bigl(\hat{s}_{t-1} + \mathrm{round}(p_t)\bigr)
$$

理想 FP32 累加 $s_t = s_{t-1} + p_t$。每步引入 $\epsilon_t = \hat{s}_t - s_t$，一般 **有偏、可累积**。当 $t$ 从 1 到 $K$：

$$
\hat{s}_K - s_K \approx \sum_{t=1}^{K} \epsilon_t
$$

$K$ 越大（MoE FFN、宽 MLA 投影），$|\hat{s}_K - s_K|$ 越容易 **肉眼影响 loss**。这就是表格里「**大量 FP8 MMA 后 partial sum 漂移**」的含义。

---

## 5. V3 对策：每 $N_c = 128$ 提升到 FP32

Figure 7(b) **Increasing accumulation precision**：

1. Tensor Core 内用 **低精度 acc** 收 partial sum（快）；
2. **每累计 $N_c = 128$ 个 MMA 元素**，把当前 partial **flush 到 CUDA Core 的 FP32 寄存器**，以 FP32 续加；
3. TC 侧 acc **清零**，再收下一段 128 个 MMA。

效果：低精度链最长只有 **128 步**，误差被 **周期性「归零到 FP32」**；总内积仍由多段 FP32 和组成，**漂移上界** 与「全程 TC 低精度累加 $K$ 步」相比大幅缩小。

对应专文一句：**算在 TC、稳在 CUDA FP32**。

---

## 6. 和推理 FP8 的区别

| | **训练（本文 / V3 Figure 7）** | **推理 FP8（如 draft benchmark）** |
|---|-------------------------------|-------------------------------------|
| 目标 | 671B **可收敛** 预训练 | **延迟 / 吞吐** |
| partial sum | 显式 **FP32 promotion** 内核 | 依引擎实现，未必同策略 |
| 块 scale | 训练侧 dynamic $s_x, s_w$ | 部署 calibrate / 静态 scale 常见 |

推理里「FP8 更快」不自动等于训练里同一套 anti-drift；读 partial sum 漂移应 **锚定 V3 训练 GEMM**。

---

## 7. 一句话

**Partial sum 漂移** = FP8 MMA 在 Tensor Core 里 **太长链低精度累加** 导致的系统性数值偏差；V3 用 **每 128 MMA 提升到 FP32** 截断累加链，与块级 dynamic scale **互补**，共同把 FP8 训练做稳。
