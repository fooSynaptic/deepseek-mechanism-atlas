# Step 6 上下文门控：依据与「记忆依赖过滤」

[← 返回 Step 6](../../../../../07-Engram/02-Engram系列导读.md#step-6-上下文门控) · [前向总图 01c](../diagrams/engram-01c-forward-dataflow.svg) · [答疑目录](README.md)

---

## 问题

[01c 前向图](../diagrams/engram-01c-forward-dataflow.svg) 里 Step 6 标成「过滤记忆」——门控 $\alpha_t$ 的**设计依据**是什么？为什么说 gate 是 **记忆依赖** 的过滤，而不是普通标量门？

---

## 结论（先答）

| 维度 | Step 1–5 | Step 6 门控 |
|------|----------|-------------|
| $e_t$ / $k_t,v_t$ | **只依赖 input n-gram**，与当前句义无关 | 用 **$h_t$（语境）× $k_t$（本条记忆）** 算 $\alpha_t$ |
| 输出 | 静态查表结果 | **$\tilde{v}_t=\alpha_t v_t$**：本条记忆留多少 |
| 角色 | O(1) 检索 | **动态过滤**：碰撞 / 歧义 / 不合语境 → $\alpha_t\to 0$ |

**依据**：查表给的是「这个 n-gram **通常** 伴随什么向量」，不是「**此刻** 该不该用」；必须用已含上下文的 $h_t$ 与 **本条** 记忆的 $k_t$ 做对齐，才能决定 $v_t$ 是否注入 backbone。这就是 **记忆依赖**——$\alpha_t$ 随 **检索到的记忆内容**（经 $W_K$）与 **当前 hidden**  jointly 变化，而非每位置固定系数。

### α_t 算「一次 Attention」吗？

**可以说：像 Attention 的「一个 Q 对一个 K」的那一步**；**不能说：等于一层 Self-Attention**。

| | 标准 Attention（位置 $t$） | Engram Step 6 的 $\alpha_t$ |
|--|---------------------------|------------------------------|
| Q·K | $\mathrm{softmax}_j\bigl(q_t^\top k_j\bigr)$，对 **多个** $j$ 归一化 | **一个** $h_t$ 对 **一条** $k_t$，无对 $j$ 的 softmax |
| 非线性 | softmax → 权重和为 1 | $\sigma(\cdot)$ → $\alpha_t\in(0,1)$，**不要求**与其他位置竞争 |
| V 聚合 | $\sum_j w_{t,j} v_j$ | **标量** $\alpha_t$ 只缩放 **本条** $v_t$：$\tilde{v}_t=\alpha_t v_t$ |
| K/V 来源 | 同一 hidden 线性投影 | K/V 来自 **查表** $e_t$ |

更贴切的说法：

- **是**：一次 **scaled dot-product 兼容性打分**（Q=$h_t$，K=$k_t$），再乘 V=$v_t$ —— 和 Attention 核心算子同族。
- **不是**：对序列 / 多记忆槽的 **分布归一化注意力**；没有「在 $L$ 个或 $K$ 个头里选谁」那一步（多头哈希的 $K$ 个头已在 Step 4 **拼进** $e_t$，Step 6 是对整段 $e_t$ 投影后的 **一个** 门）。

口诀：**「单键 attention 打分 + sigmoid 门」**，不是 **「一层 multi-head self-attention」**。后面 backbone 里的 **Attention** 仍负责全局依赖；Step 6 只决定 **这条静态记忆此刻放多少进来**。

---

## 1. 为什么 Step 1–5 不能代替门控

Step 1–4 产出 **上下文无关** 的 $e_t$：

$$
e_t = \Big\Vert_{n,k} E_{n,k}[\varphi_{n,k}(g_{t,n})]
$$

索引 $\varphi_{n,k}(g_{t,n})$ **只依赖 input token**，因此：

1. **哈希碰撞**：不同 n-gram 映射到同一槽位 → 表项是混合先验，未必对应当前片段。
2. **多义 / 歧义**：同一表面 n-gram 在不同句子里语义不同（图脚注「Apple vs apple」类问题在 Step 1 部分缓解，但不能消歧整句）。
3. **静态先验 vs 动态需求**：$e_t$ 回答「历史上这个搭配常出现什么表示」，不回答「**这一句** 现在是否需要这条先验」。

Step 5 把 $e_t$ 投影为 $k_t, v_t$，仍 **未使用** $h_t$——只是「把记忆装进 K/V 形态」，**还没有过滤**（01c 图：备过滤）。

---

## 2. 门控在算什么（与 Attention 的异同）

**论文 / overview 写法**（Scaled dot-product + sigmoid）：

$$
\alpha_t = \sigma\left(\frac{\mathrm{RMSNorm}(h_t)^{\top} \mathrm{RMSNorm}(k_t)}{\sqrt{d}}\right), \quad \tilde{v}_t = \alpha_t \cdot v_t
$$

| 角色 | 张量 | 语义 |
|------|------|------|
| **Query** | $h_t$ | 当前层 hidden，已吸收前文 Attention / 下层信息 → **此刻语境** |
| **Key** | $k_t = W_K e_t$ | **本条** 查表记忆在「对齐空间」的表示 |
| **Value** | $v_t = W_V e_t$ | 若通过门控，实际写入记忆支路的内容 |
| **门** | $\alpha_t \in (0,1)$ | $h_t$ 与 $k_t$ **越一致** 越大；不合 → 趋近 0 |

**与 Attention 相同点**：用归一化 Q·K 衡量 **兼容性**（scaled dot product）。

**与 Attention 不同点**：

| | Attention | Engram Step 6 |
|--|-----------|----------------|
| K/V 来源 | 由 hidden **算出来** | K/V 来自 **查表** $e_t$（O(1)） |
| 范围 | 可对全序列做 softmax | **标量** $\alpha_t$ 门控本条记忆（非全局分布） |
| 计算 | $O(L^2)$ 或稀疏变体 | 查表 O(1) + 一次 dot + sigmoid |

本质：**Attention 式对齐**，但 **记忆是条件查表来的**，所以叫 *context-aware gating* / *记忆过滤*。

---

## 3. 为何叫「记忆依赖」的过滤

「记忆依赖」指 $\alpha_t$ **不是** 只由位置 $t$ 或只由 $h_t$ 决定，而是 **耦合本条检索结果**：

$$
\alpha_t = f\bigl(h_t,\; k_t(e_t(g_{t,\cdot}))\bigr)
$$

- 换一句 input → $g_{t,n}$ 变 → $e_t$ 变 → $k_t, v_t$ 变 → **即使 $h_t$ 维数相同，$\alpha_t$ 也可完全不同**。
- 同一句里不同层 / 不同分支：$h_t$ 变 → 对 **同一条** $e_t$ 也可得到不同 $\alpha_t$（语境变了，是否采纳这条静态先验也变）。

因此 gate 过滤的是 **「这一条记忆」对当前语境的贡献**，不是对 hidden 做通道级 LayerScale 那种与查表无关的门。

**为何乘在 $v_t$ 上**：注入 backbone 的是 value 支路；$k_t$ 只参与 **是否放行**，$v_t$ 是 **放行后的内容**。Step 7–8 只在 $\tilde{v}_t$ 上运算（01c 图例：S7–8 不再接触未过滤 $v_t$）。

---

## 4. 官方 demo 实现（与公式的关系）

[engram_demo_v1.py](../../../../../07-Engram/engram_demo_v1.py) 中 `Engram.forward`（约 L358–377）：

```python
key = self.key_projs[hc_idx](embeddings)
normed_key = self.norm1[hc_idx](key)
query = hidden_states[:, :, hc_idx, :]
normed_query = self.norm2[hc_idx](query)
gate = (normed_key * normed_query).sum(dim=-1) / math.sqrt(hidden_size)
gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()  # demo 额外：保符号的 sqrt
gate = gate.sigmoid().unsqueeze(-1)
value = gates * self.value_proj(embeddings).unsqueeze(2)
```

| 项 | demo | overview 公式 |
|----|------|----------------|
| 归一化 | `norm1` on key，`norm2` on query | $\mathrm{RMSNorm}(k_t)$，$\mathrm{RMSNorm}(h_t)$ |
| 对齐分数 | 逐维积再 `sum` = 内积 | $h_t^\top k_t$ |
| 缩放 | `/ sqrt(hidden_size)` | `/ sqrt{d}$ |
| 非线性 | `signed_sqrt` 再 `sigmoid` | 直接 $\sigma(\cdot)$ |
| 多分支 | `hc_mult` 路独立 gate | overview §多分支有 $\alpha_t^{(m)}$ |

**阅读建议**：结构一致（**normed Q·K → 门 → × value**）；demo 在 sigmoid 前多一步 signed sqrt，属实现细节，不改变「用 $h_t$ 与 **本条 memory key** 决定是否采用 $v_t$」这一设计。

---

## 5. 若去掉 Step 6 会怎样（直觉）

| 有门控 | 无门控（$\alpha_t\equiv 1$） |
|--------|------------------------------|
| 碰撞 / 歧义表项可被压掉 | 噪声 $v_t$ 直接进 Conv / 残差 |
| 静态先验 **按需** 接入 | 所有 n-gram 查表结果 **强制** 影响 hidden |
| 与论文「条件记忆」一致 | 退化成「查表 + 固定加回」，难扛多义 |

Engram-Nine 等工作进一步讨论 **gating credit assignment**（门控是否跟对表项）；瓶颈常在 **门控质量** 而非索引精度（见 overview §热冷层 / Nine 梗概）。

---

## 6. 与前后 Step 的衔接（对照 01c）

```
S1–4: e_t 静态（input only）
S5:   k_t, v_t = W_K e_t, W_V e_t   ← 备过滤
S6:   α_t = f(h_t, k_t);  ṽ_t = α_t v_t   ← ★ 记忆过滤
S7–8: 只用 ṽ，写回 H
```

→ **短卷积 / 感受野**：[step7-short-conv-receptive-field.md](step7-short-conv-receptive-field.md)（在 **已过滤** 的 $\tilde{V}$ 上扩序列混合）

---

## 参考

- [engram-series-overview.md §Step 5–6](../../../../../07-Engram/02-Engram系列导读.md#阶段-b上下文门控与融合)
- [engram-01c-forward-dataflow.svg](../diagrams/engram-01c-forward-dataflow.svg)
- [engram_demo_v1.py](../../../../../07-Engram/engram_demo_v1.py) — `Engram.forward`
