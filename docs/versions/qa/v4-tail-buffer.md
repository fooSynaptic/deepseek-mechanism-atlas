# V4 里的 Tail buffer 是什么？

[← 返回 CSA/HCA 一句话](../csa-hca-mixed-attention.md#一句话) · [§4 异构 KV](../csa-hca-mixed-attention.md#v4-mixed-attention) · [KV Layout §State](../v4-kv-layout.md#state-cache) · [磁盘 Prefix §可落盘对象](../v4-disk-prefix-cache.md#哪些-kv-能落盘共享) · [答疑目录](./README.md)

---

## 一句话

**Tail buffer** 是流式 decode 时 **尚未凑满一块** 的 token 尾（CSA 缺 **4**、HCA 缺 **128**）：在 **State cache** 里以 **未压缩** K/V 暂存，**凑满后** 才压成 entry **append 进 Classical 池**——是压缩算法的 **块对齐缓冲**，不是第四种 attention 机制。

---

## 1. 为何需要 tail

块压缩要求 **连续 $m$ 个 token** 才能输出一条 entry：

| 路径 | 块大小 $m$ | Tail 含义 |
|------|------------|-----------|
| **CSA** | $m{=}4$ | 当前序列长度 $\bmod 4 \neq 0$ 的 **1～3** 个尾 token |
| **HCA** | $m'{=}128$ | 不足 128 的 **未压缩尾** |

若强行写入 Classical 池会破坏 **$\mathrm{lcm}(4,128)$ 对齐**（[KV layout §Classical](../v4-kv-layout.md#classical-kv-cache)），故 tail **单独挂在 State 池**，与 SWA 同属 **「仍在形成中」** 的状态。

---

## 2. 生命周期

| 阶段 | Tail | Classical 池 |
|------|------|--------------|
| Prefill / 流式 append | 新 token 先 **进 tail** | 仅 **已满块** 写入 |
| Tail 达到 $m$ | **触发压缩** → 一条新 entry | **append** 不可变 entry |
| Prefix 共享 | Tail **随请求推进变化** | 已满块可 **共享 / 落盘** |

因此 [磁盘 Prefix Cache](../v4-disk-prefix-cache.md) 对 tail 的结论是：**通常随请求走 State 池**，不像 CSA/HCA 压缩块那样 **直接 immutable 落盘**。

---

## 3. 与 SWA 同池、分工不同

| | **Tail buffer** | **SWA** |
|--|-----------------|---------|
| 内容 | **等待压缩** 的原始 token K/V | **窗口内精确局部** K/V |
| 是否可变 | 每来一 token **可能变长/清空** | **滑动更新** 窗口 |
| 算法角色 | **块对齐工程缓冲** | **注意力精度** 保障 |

二者都在 **State cache**，但 eviction、prefix 策略 **[各自独立](../v4-kv-layout.md#state-cache)**。

---

## 4. 读图提示

[异构 KV 图](../../figures/v4/v4-hetero-kv.svg) 下半 **State / tail** 区：token 流在进 CSA 4:1 / HCA 128:1 **之前**，末尾不足一块的部分 **不会** 出现在 Classical 条带里。

---

## 5. 相关阅读

| 文档 | 内容 |
|------|------|
| [V4 KV Layout](../v4-kv-layout.md) | Classical vs State 分工 |
| [CSA / HCA 混合压缩注意力§4](../csa-hca-mixed-attention.md#v4-mixed-attention) | 五类对象对照表 |
