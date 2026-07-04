# 名词解释：MMA

[← 返回 FP8 专文](../v3-fp8-dynamic-quantization.md) · [partial sum 漂移](./fp8-partial-sum-drift.md) · [答疑目录](./README.md)

---

## MMA 是什么

**MMA** = **M**atrix **M**ultiply-**A**ccumulate（矩阵乘加）。

一次 MMA 在硬件上做：

$$
D \leftarrow A \times B + D
$$

即：取两块小矩阵（或向量块）相乘，结果 **加进** 累加器 $D$（不是每次写新内存）。大矩阵乘 **GEMM** 在 GPU 上就是 **成千上万次 MMA 拼出来** 的。

---

## 在 DeepSeek-V3 FP8 语境里指什么

| 词 | 含义 |
|----|------|
| **MMA** | Tensor Core 上的 **单次/单步** 低精度乘加（V3 训练用 **FP8**） |
| **GEMM** | 整层线性层 $Y=XW$ 的 **完整矩阵乘**（内部 = 多轮 MMA） |
| **WGMMA** | **W**arp-**G**roup **MMA**；Hopper（H100 等）上 **一组 warp 协作** 的一条 MMA 指令流，V3 论文 Figure 7(b) 标成 WGMMA 1–4 |
| [**partial sum** | 某一输出元素在 **整条 MMA 链还没加完** 时的中间累加和](./fp8-partial-sum-drift.md) |

V3 文档里「**每 $N_c{=}128$ 次 MMA**」= 每做满 **128 个 MMA 粒度的乘加贡献**，就把 Tensor Core 里低精度 partial sum **提到 CUDA Core FP32** 再累，防止漂移。

---

## 和 MHA / MQA 不要混

| 缩写 | 领域 | 含义 |
|------|------|------|
| **MMA** | **GPU 算子 / FP8 训练** | Matrix Multiply-Accumulate |
| **MHA** | **注意力** | Multi-**H**ead **A**ttention |
| **MQA** | **注意力** | Multi-**Q**uery **A**ttention |

V3.1 **Prefill MHA / Decode MQA** 是 MLA **模式切换**，与 FP8 的 **MMA 指令** 无关。

---

## 硬件落点

| 硬件 | MMA 相关职责 |
|------|----------------|
| **Tensor Core** | 执行 FP8 **MMA / WGMMA**；块内低精度乘加、短链 partial sum |
| **CUDA Core** | MMA 结果 **反量化**（乘 $s_x s_w$）；partial sum **FP32 续累加** |

---

## 一句话

**MMA** = GPU Tensor Core 的 **矩阵乘加原语**；V3 FP8 训练里的「MMA 漂移 / 每 128 MMA 提升 FP32」都是在说 **这条乘加累加链** 的数值行为，不是注意力里的 MHA/MQA。
