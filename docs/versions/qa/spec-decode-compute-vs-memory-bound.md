# 投机解码：Compute-Bound vs Memory-Bound — DFlash / Eagle 如何对应？

[← 返回投机解码专文 §1.1](../dspark-speculative-decoding.md#11-瓶颈每-token-搬一遍大模型权重) · [← 专文 §4 草稿范式](../dspark-speculative-decoding.md#4-草稿范式总览eagle3--dflash--mtp--dspark) · [← 酱紫君解读 §Speculative Decoding](../../reports/zhihu-jiangzijun-dspark-highlights-20260627.md#speculative-decoding) · [答疑目录](./README.md)

酱紫君原文一句话：

> 用 **Compute-Bound 的并行验证** 去替代 **Memory-Bound 的串行读取**。

下文分 **两层** 读这句话，并说明 **DFlash / Eagle** 在 draft 侧如何对应。

---

## 1. 两层含义：不要混在一个维度上

| 层级 | 「前者」Compute-Bound 并行 | 「后者」Memory-Bound 串行 | 代表 |
|------|---------------------------|--------------------------|------|
| **A. 整链宏观**（target 侧） | target **一次前向**并行校验 $K$ 个位置 | 传统 decode：每 token **各跑 1 次** target 前向 | §1.1–§1.2 |
| **B. Draft 范式**（$M_q$ 侧） | **并行**出整块 $K$ 位 hidden / logits | **串行**自回归，每步完整 draft 前向 | **DFlash** vs **Eagle3** |

**结论（直接回答）**：

- 若问的是 **draft 怎么猜 $K$ 个 token**：**DFlash ≈ 前者（并行 / 更偏 Compute-Bound）**，**Eagle3 ≈ 后者（串行 / Memory-Bound）**。
- 若问的是 **酱紫君开篇整句在说什么**：主语是 **投机解码相对纯自回归**——用 target 的 **1 次并行 verify** 换掉 **$K$ 次串行 target decode**；Eagle / DFlash 是这之后 **draft 侧的两条工程路线**，不是整句的字面主语。

两层 **同构**：都是「**少搬几次权重，多算一点并行位**」。

---

## 2. 为何 decode 默认是 Memory-Bound？

专文 §1.1 的物理图景（batch=1 自回归）：

1. 每生成 1 个 token，都要把 **整网权重** 从 **HBM** 搬进 **SM**（[名词：SM](./gpu-sm-term.md)；大矩阵 × 单向量，**GEMV**）。
2. 算术强度极低 → GPU 算力闲置，时间花在 **访存带宽** 上 → **Memory-Bound**。
3. 第 $N$ 个 token 必须等第 $N{-}1$ 个 → **串行读取**（每步一遍权重流）。

传统 target decode 出 $K$ 个 token：**$K$ 次** target 前向 ≈ **$K$ 次**搬大模型权重。

投机解码的 target **verify**：仍搬 **1 遍** target 权重，但 attention / FFN 在 **$K$ 个位置并行算**——多出来的 FLOPS 相对带宽往往 **填不满 SM**（[名词：SM](./gpu-sm-term.md#搬进-smvs-饱和-sm)），但 **不再乘 $K$ 遍搬权重** → 从系统视角 **更像 Compute-Bound 的并行验证**（相对「$K$ 次 mem-bound 串行读」）。

---

## 3. 为何 Eagle3 是 Memory-Bound 串行？

Eagle3 属于 **自回归外挂 draft**：块内第 $i$ 个 token 依赖已猜的 $\hat{x}_{t+i-1}$，**不能**与前面位完全并行，必须：

$$
\hat{x}_{t+1} \to \text{draft 前向}_1 \to \hat{x}_{t+2} \to \text{draft 前向}_2 \to \cdots \to \hat{x}_{t+K}
$$

### 3.1 每步都在「再读一遍 draft 权重」

与 target 同理，draft 在 batch=1 decode 下也是 **GEMV 型** 前向。Eagle3 的第 $i$ 步 = **一整段** draft Transformer（多层 Attention + FFN/MoE）完整前向 → 从 HBM **再拉一遍** draft 全部层权重。

- 块长 $K$ → **$K$ 次**完整 draft 前向
- draft 侧总耗时 $\tau_q^{\mathrm{Eagle}} \approx K \cdot \tau_{\mathrm{step}}$ → **$\tau_q \propto K$**（专文 §4、酱紫君解读对照表）

### 3.2 算力利用率低

每步只推进 **1 个 token** 的 hidden；矩阵乘法是「大矩阵 × 单向量」，SM 上并行度有限。时间轴上大量周期在 **等 HBM 返回下一层权重**，而非饱和 Tensor Core → 典型 **Memory-Bound**。

### 3.3 换来的收益：块内因果对齐

串行的代价是 **每步显式看见** $\hat{x}_{t+i-1}$，draft 分布 $q_i$ 更贴近 target 校验分布 $p_i$ → $\mathbb{E}[N_{\mathrm{acc}}]$ **高、后缀稳定**（专文 §6.1 表）。Eagle 用 **多搬几次权重** 换 **猜得更准**。

---

## 4. 为何 DFlash 更偏 Compute-Bound 并行？

DFlash 走 **并行外挂 draft**：在固定 target hidden $h_t^p$ 上，**一次前向**同时产出 $K$ 个位置的 hidden / 基础 logits：

$$
E = \{e_{t+1}, \ldots, e_{t+K}\} = \mathrm{DFlashBackbone}(h_t^p)
$$

### 4.1 权重只搬一轮，$K$ 位并行 matmul

- **2–3 层**轻量 MoE：**1 次** HBM 权重流 → **$K$ 个位置**并行做 Attention / FFN。
- 相对 Eagle 的 $K$ 次完整前向，**访存次数 ~$1/K$**；多出来的 FLOPS 用于 **并行位** 而非 **重复搬权重** → draft 延迟 **几乎与 $K$ 弱相关**（专文 §4 表「DFlash 优势」）。

### 4.2 瓶颈从带宽转向算力

同一轮 draft 里，SM 处理 **$K$ 宽**的激活 → 算术强度高于「$K$ 次单向量 GEMV」。在固定 draft 参数量下，**更吃算力、更少重复访存** → 工程上归类为 **Compute-Bound 并行**一侧。

### 4.3 代价：缺块内因果

各位 **共享** target 前缀，**互不见**块内已猜 $\hat{x}_{<i}$ → 第 2 位起 $q_i \not\approx p_i$ → $\mathbb{E}[N_{\mathrm{acc}}]$ **骤降**（专文 §6.1）。DFlash 用 **少搬权重、并行算** 换 **低 draft 延迟**，但 **后缀接受率** 不如 Eagle。

---

## 5. DSpark：在同一轮 draft 里拆「重并行 + 轻串行」

DSpark 不是第三条 mem/compute 轴，而是 **把 Eagle 的串行依赖拆成极轻顺序头**：

| 阶段 | 访存 / 算力 | 作用 |
|------|-------------|------|
| **并行主干**（DFlash 改进） | 1 次 HBM 拉 MoE；$K$ 位并行 | 低 $\tau_q$、第 1 位高接受率 |
| **顺序头** $g_\theta$ × $K$ | **寄存器 / on-chip**；$<0.1\%$ 主模型参数 | 逐位注入 $\hat{x}_{i-1}$，补块内因果 |

若顺序头也每层跑 MoE → **退化成 Eagle**，$\tau_q$ 再次 $\propto K$（[酱紫君解读 §半自回归](../../reports/zhihu-jiangzijun-dspark-highlights-20260627.md#dspark-半自回归草稿并行主干-vs-顺序头)）。

---

## 6. 对照总表

| | **Eagle3**（后者 / 串行） | **DFlash**（前者 / 并行） | **DSpark** |
|--|--------------------------|--------------------------|------------|
| Draft 生成 | $K$ 次完整 Transformer 前向 | **1 次**并行主干 | 1 次主干 + $K$ 次轻头 |
| 主导瓶颈 | **HBM 带宽**（重复搬 draft 权重） | **并行算力**（$K$ 宽 matmul） | 主干同 DFlash；轻头 on-chip |
| $\tau_q$ vs $K$ | **$\propto K$** | **≈ 常数** | 主干 ≈ 常数 + 极小尾巴 |
| $\mathbb{E}[N_{\mathrm{acc}}]$ | **高**（块内因果完整） | 第 1 位高，**后缀掉** | 折中 |
| 加速公式 $S_{\uparrow}$ 杠杆 | 抬接受率；**$K\tau_q$ 大** | **$K\tau_q$ 小**；后缀接受差 | 两者折中 |

加速比粗式（专文 §4）：

$$
S_{\uparrow} = \frac{\bigl(\mathbb{E}[N_{\mathrm{acc}}] + 1\bigr)\,\tau_p}{K\,\tau_q + \tau_p}
$$

- Eagle：**分子大**（$\mathbb{E}[N_{\mathrm{acc}}]$），**分母里 $K\tau_q$ 也大**
- DFlash：**分母小**，但 $\mathbb{E}[N_{\mathrm{acc}}]$ 限制分子
- DSpark：在 **$K\tau_q$** 与 **$\mathbb{E}[N_{\mathrm{acc}}]$** 之间取半自回归折中

---

## 7. 一句话记忆

- **Memory-Bound 串行读取** = batch=1 decode 下 **每 token 各搬一遍权重**；Eagle draft 是典型（$K$ token → $K$ 遍 draft 权重流）。
- **Compute-Bound 并行验证** = **搬一遍权重、多位置并行算**；target verify 与 DFlash 并行主干同属这一类。
- **DFlash ≈ 前者，Eagle ≈ 后者** —— 在 **draft 范式**维度上成立；开篇整句 additionally 指 **target 并行 verify 替代 $K$ 次串行 target decode**。

---

## 8. 反向引用

| 文档 | 锚点 / 说明 |
|------|-------------|
| [投机解码专文 §1.1](../dspark-speculative-decoding.md#11-瓶颈每-token-搬一遍大模型权重) | Memory-Bound 瓶颈原文 |
| [投机解码专文 §4](../dspark-speculative-decoding.md#4-草稿范式总览eagle3--dflash--mtp--dspark) | Eagle / DFlash / DSpark 范式表 |
| [投机解码专文 §6.1](../dspark-speculative-decoding.md#61-半自回归候选生成) | 堆叠依赖与接受率 |
| [酱紫君解读 §Speculative Decoding](../../reports/zhihu-jiangzijun-dspark-highlights-20260627.md#speculative-decoding) | 「用并行验证替代串行读取」原文 |
| [酱紫君解读 §半自回归](../../reports/zhihu-jiangzijun-dspark-highlights-20260627.md#dspark-半自回归草稿并行主干-vs-顺序头) | Eagle 式 vs DSpark 式 $\tau_q$ 对照 |
| [投机解码：为何接受率是 $\min$、修正分布是 $\max$](./spec-decode-rejection-sampling.md) | lossless 接受路径（与 mem/compute 正交） |
| [名词解释：SM](./gpu-sm-term.md) | SM / HBM / 「填不满 SM」名词 |

## 9. 外部文献

- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, arXiv:2211.17002, 2022.
- Li et al., *EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*, 2024（Eagle 系自回归 draft）。
- [DSpark_paper.pdf](https://github.com/deepseek-ai/DeepSpec/blob/main/DSpark_paper.pdf) — Semi-Autoregressive Draft；DFlash 并行 vs Eagle 串行消融。
