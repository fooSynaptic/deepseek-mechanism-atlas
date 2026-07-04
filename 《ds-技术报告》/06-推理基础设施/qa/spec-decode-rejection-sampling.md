# 投机解码：为何接受率是 $\min$、修正分布是 $\max$

[← 返回投机解码专文 §1.3](../04-DSpark投机解码.md#13-净收益lossless-与最坏情况) · [答疑目录](../../01-总览/qa/README.md)

专文 §1.3 仅保留结论；本节用引用块展开 **接受路径 / 拒绝路径 / 合并为 $p$** 与公式推导。

> **接受路径** — 固定前缀 $s$，draft 抽 $X \sim q$，再独立抽 $U \sim \mathrm{Uniform}(0,1)$。对 token $x$，事件 $E_{\mathrm{acc}}(x)=\{X{=}x,\ U\le\alpha(x)\}$：draft **恰好提议 $x$** 且 **校验通过**，直接输出 $x$。
>
> $$P\bigl(E_{\mathrm{acc}}(x)\bigr)=P(X{=}x)\cdot P(U\le\alpha(x)\mid X{=}x)=q(x)\,\alpha(x).$$
>
> 连乘是序贯试验的乘法公式：先以 $q(x)$ 抽到 $x$，再以 $\alpha(x)$ 通过校验。

> **拒绝路径** — 「发生拒绝且最终输出 $x$」：丢弃提议的 $X$，从 $p_{\mathrm{resample}}$ 抽到 $X'{=}x$。与 $E_{\mathrm{acc}}(x)$ **不交**。

> **合并为 target 分布** — 输出 $x$ 的总概率
>
> $$\pi(x)=P\bigl(E_{\mathrm{acc}}(x)\bigr)+P(\mathrm{reject})\,p_{\mathrm{resample}}(x).$$
>
> 取 $\alpha(x)=\min(1,p(x)/q(x))$ 使接受路径贡献 $q\alpha=\min(q,p)$；取 $p_{\mathrm{resample}}(x)\propto\max(0,p(x)-q(x))$ 补足拒绝路径上的剩余质量，可得 $\pi(x)=p(x)$（§5 逐步验算）。draft 只改 $q$ 与接受有多快，**不改**边际分布。

---

## 1. 问题设定

固定已接受前缀 $s$。希望下一个 token 服从 **target 分布** $p(\cdot)\equiv p_{\mathrm{tgt}}(\cdot \mid s)$。

投机解码让 **draft** 先提议 token $x$，其（条件）概率为 $q(x)\equiv p_{\mathrm{draft}}(x \mid s)$。算法只做两件事：

1. 以某个概率 **接受** $x$；
2. 否则 **拒绝**，再从另一个分布 **重采** $x'$。

要证明 **lossless**，只需证明：对任意词汇表元素 $x$，最终输出 $x$ 的总概率 $\pi(x)$ 恰好等于 $p(x)$。

---

## 2. 单步随机实验：何谓「接受路径」、为何是连乘

### 2.1 算法对应的一次抽样

把「draft 提议 → target 校验」看成 **两步随机试验**：

1. **提议**：随机变量 $X \sim q$（draft 按 $q$ 抽一个 token，$P(X{=}x)=q(x)$）。
2. **校验**：再抽 $U \sim \mathrm{Uniform}(0,1)$，与 $X$ 独立；若 $U \le \alpha(X)$ 则 **接受**，否则 **拒绝**。

其中 $\alpha(x)$ 是待定的接受率（下一节解出 $\alpha(x)=\min(1,p(x)/q(x))$）。

**输出规则**：

| 事件 | 输出 |
|------|------|
| 接受（$U \le \alpha(X)$） | 直接输出提议的 $X$ |
| 拒绝（$U > \alpha(X)$） | 丢弃 $X$，从 $p_{\mathrm{resample}}$ **另抽** $X'$ 输出 |

### 2.2 「接受路径」的严格含义

对某个 **具体的** token $x$，定义事件

$$
E_{\mathrm{acc}}(x) \;=\; \{ X = x \;\land\; U \le \alpha(x) \}.
$$

即：**draft 恰好提议了 $x$，且校验通过**。全文所称 **接受路径（acceptance path）** 就是指输出 $x$ 时，$E_{\mathrm{acc}}(x)$ 发生这一条分支。

与之互斥的是 **拒绝路径（rejection path）**：

$$
E_{\mathrm{rej}}(x) \;=\; \{ \text{发生拒绝} \} \;\land\; \{ X' = x \},
$$

其中 $X' \sim p_{\mathrm{resample}}$ 是拒绝后另抽的 token。

两条路径 **不交**：接受时输出的是 $X$，拒绝时输出的是 $X' \neq X$（逻辑上先拒绝再重采，不会「又接受 draft 又重采」）。

### 2.3 为何是 $q\,\alpha$

由 **乘法公式 / 条件概率**（$U$ 与 $X$ 独立，故给定 $X{=}x$ 时接受概率为 $\alpha(x)$）：

$$
P\bigl(E_{\mathrm{acc}}(x)\bigr)
= P(X{=}x) \cdot P(U \le \alpha(x) \mid X{=}x)
= q(x)\,\alpha(x).
$$

**连乘不是额外假设**，而是「先以 $q(x)$ 抽到 $x$，再以 $\alpha(x)$ 通过校验」这一 **序贯试验** 的自然写法。类比：先掷骰得到 6 的概率是 $\frac{1}{6}$，再独立掷硬币正面才采纳，则「出 6 且采纳」$=\frac{1}{6}\times\frac{1}{2}$。

### 2.4 两条路径合并

任意 $x$ 的最终输出概率是 **两条不交路径之和**：

$$
\pi(x)
= \underbrace{P\bigl(E_{\mathrm{acc}}(x)\bigr)}_{q(x)\alpha(x)}
+ \underbrace{P(\mathrm{reject})\cdot p_{\mathrm{resample}}(x)}_{\text{拒绝后重采}}.
$$

后文先定 $\alpha$ 使 $q\alpha=\min(q,p)$，再定 $p_{\mathrm{resample}}$ 补足 $\max(0,p-q)$，即得 $\pi(x)=p(x)$。

---

## 3. 接受率为何是 $\min\!\left}{q}\right)$

记接受概率为 $\alpha(x)$。由 §2.3，**接受路径**上输出 $x$ 的概率为 $q(x)\,\alpha(x)$。要求这条路径对 $x$ 的贡献是 $p(x)$ 的一部分，且**不超过** $p(x)$（超出部分留给拒绝路径）。最自然要求是

$$
q(x)\,\alpha(x) = \min\bigl(q(x),\, p(x)\bigr).
$$

当 $q(x)>0$ 时两边同除以 $q(x)$：

$$
\alpha(x) = \min\!\left(1,\ \frac{p(x)}{q(x)}\right).
$$

**直觉**：

| 关系 | 含义 | 接受率 |
|------|------|--------|
| $q(x) > p(x)$ | draft **高估** $x$ | $\dfrac{p}{q} < 1$，经常拒绝以 **压低** $x$ 出现频率 |
| $q(x) \le p(x)$ | draft **低估或未覆盖** $x$ | 接受率 $=1$；$p-q$ 的正差留给重采分布补足 |

这就是 $\min(1,p/q)$ 的来源：**不是经验公式，而是要求「接受路径贡献 $\min(q,p)$」的代数解**（见 §2.2 对接受路径的定义）。

---

## 4. 修正分布为何 $\propto \max$

对每个 $x$，target 需要总质量 $p(x)$。接受路径已给出 $\min(q(x),p(x))$，还差

$$
p(x) - \min\bigl(q(x),p(x)\bigr) = \max\bigl(0,\, p(x) - q(x)\bigr).
$$

把所有 token 上「尚未满足的质量」收集起来，归一化，定义 **拒绝后重采分布**：

$$
p_{\mathrm{resample}}(x) = \frac{\max\bigl(0,\, p(x) - q(x)\bigr)}{Z}, \qquad Z = \sum_{x'} \max\bigl(0,\, p(x') - q(x')\bigr).
$$

**直觉**：$p-q>0$ 的位置，正是 draft **没给够** 的概率质量；拒绝后专门从这些位置补抽，把 target 分布「缺的那一块」填回去。

---

## 5. 合并后恰好等于 $p$

> **基础恒等式** — 记 $a=p(x)$、$b=q(x)$，则
>
> $$\min(a,b) + \max(0,\, a-b) = a.$$
>
> 分两种情形验证：若 $p \le q$，则 $\min(p,q)=p$、$\max(0,p-q)=0$，相加得 $p$；若 $p > q$，则 $\min(p,q)=q$、$\max(0,p-q)=p-q$，相加得 $q+(p-q)=p$。因此对**每个** token，接受路径质量 $\min(q,p)$ 与拒绝路径补足 $\max(0,p-q)$ **刚好拼回** target 的 $p(x)$，与 draft 无关。

拒绝概率

$$
P(\mathrm{reject}) = 1 - \sum_x q(x)\,\alpha(x) = 1 - \sum_x \min\bigl(q(x),p(x)\bigr) = Z.
$$

（由上文恒等式，$\sum_x \min(q,p) + \sum_x \max(0,p-q) = \sum_x p(x) = 1$，故 $P(\mathrm{reject}) = 1 - \sum_x \min(q,p) = \sum_x \max(0,p-q) = Z$。）

最终输出 $x$ 的概率：

$$
\pi(x) = \underbrace{q(x)\,\alpha(x)}_{\min(q,p)} + \underbrace{P(\mathrm{reject})\cdot p_{\mathrm{resample}}(x)}_{Z \cdot \frac{\max(0,p-q)}{Z}} = \min\bigl(q(x),p(x)\bigr) + \max\bigl(0,\, p(x)-q(x)\bigr) = p(x).
$$

因此对 **每一个** $x$ 都有 $\pi(x)=p(x)$：与「不用 draft、直接从 $p$ 采样」**同分布**。

---

## 6. 与投机解码整链的关系

- 上面对 **单个位置、固定前缀 $s$** 证明一次；链上第 $1,2,\ldots$ 位在 **更新前缀后** 重复同一规则。
- target **一次并行前向** 只是同时算出各位置的 $p_{\mathrm{tgt}}$；接受/拒绝规则不变。
- draft 只影响 $q$ 与接受有多快，**不改变** $\pi=p$ 的构造。

---

## 7. 参考

- Leviathan, Y., Kalman, M., & Matias, Y. *Fast Inference from Transformers via Speculative Decoding.* arXiv:2211.17002.
- 专文机制总览：[投机解码与 DSpark§1.2–§1.3](../04-DSpark投机解码.md#12-一轮怎么走draft--verify--accept)
