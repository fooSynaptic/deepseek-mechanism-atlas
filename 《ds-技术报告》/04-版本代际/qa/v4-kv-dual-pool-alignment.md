# V4 为何要分池？块对齐与尾缓冲怎么配合？

[← 返回 KV Layout §为何单一 layout 不够](../../06-推理基础设施/05-V4-KV-Layout.md#为什么-v3v32-的单一-layout-不够) · [Classical 池](../../06-推理基础设施/05-V4-KV-Layout.md#classical-kv-cache) · [State 池](../../06-推理基础设施/05-V4-KV-Layout.md#state-cache) · [Tail buffer](v4-tail-buffer.md) · [SWA](v4-swa-sliding-window.md) · [答疑目录](../../01-总览/qa/README.md)

---

## 一句话

V4 同时有 **两种块长**（CSA $m{=}4$、HCA $m'{=}128$）和 **两种生命周期**（已压缩定型 vs 仍在滑动/凑块）；**Classical 池** 只放 **凑满且不可变** 的压缩 entry，**State 池** 放 **尾缓冲 + SWA** 等 **每步都在变** 的状态——混在一个池里会同时破坏 **$\mathrm{lcm}$ 块对齐、prefix 共享、HiSparse offload**。

---

## 1. 为什么要分池管理

### 1.1 不是「多几种 KV」那么简单，是 **读写语义不同**

| | **Classical KV cache** | **State cache** |
|--|------------------------|-----------------|
| 存什么 | 已凑满的 **CSA C4 / HCA** 压缩 entry | **Tail**（未凑满块）+ **SWA**（滑动窗口精确 K/V）+ 可能含 Indexer |
| 是否可变 | 写入后 **不可变**（immutable） | **每 decode 步** 都可能变（tail 变长/清空，SWA 滑动） |
| 能否 prefix 共享 | **能** — 同一压缩块多请求只存一份 | **难** — 随各请求当前长度 / 窗口不同 |
| 能否落盘 / offload | CSA/HCA 块 **优先**（紧凑、不变） | 通常 **跟请求走**；SWA 还有三档策略（[磁盘 prefix](../../06-推理基础设施/07-V4-磁盘Prefix-Cache.md)） |
| 分配粒度 | 按 **压缩 entry** append | **每请求固定 state block**（batch 内对齐） |

V3/V3.2 可以「一层一条同质 latent 流」——**shape 一样、生命周期一样**。V4 把 **远距压缩历史** 与 **局部未压缩 / 未定型** 混在同一抽象里，引擎就无法用 **同一套 eviction、同一套 prefix 哈希、同一套 DMA 粒度** 管理。

### 1.2 压缩比 $m \neq m'$ 放大了这个问题

- **CSA**：每 **4** token → 1 条 C4 entry（远距 **稀疏** top-$k$ 读）
- **HCA**：每 **128** token → 1 条 HCA entry（全局 **dense** 读）

二者 **entry 密度差 32 倍**，读模式也不同。若强行共池：

1. **块边界对不齐** — 同一 token 区间在 CSA 是 32 条 entry、在 HCA 是 1 条，物理槽位无法统一编号；
2. **尾部长度不同** — CSA tail 最多 3 token，HCA tail 最多 127 token，与 SWA 窗口 **叠加** 后 State 侧体积不可预测；
3. **eviction 策略冲突** — HiSparse 想 offload **inactive C4**；SWA 却要 **热窗口常驻** — 必须在池级分开。

所以表里写：**$m{=}4$、$m'{=}128$ 不同 → 块对齐、尾缓冲、SWA 窗口必须分池管理**。

---

## 2. 块对齐：$\mathrm{lcm}(4,128)=128$ 是啥逻辑

### 2.1 压缩器的硬约束

块压缩不是「任意截断」，而是：

$$
\text{连续 } m \text{ 个 token 的 K/V} \;\xrightarrow{\;\text{compress}\;}\; 1 \text{ 条 entry}
$$

- CSA：$m=4$
- HCA：$m'=128$

引擎要为 **已完成的块** 在 Classical 池里分配 **连续、可索引** 的槽位。若物理分配粒度只按 4 对齐，HCA 每 128 token 才一条，中间会产生 **跨槽碎片**；若只按 128 对齐，CSA 每 4 token 一条又无法在子块内寻址。

### 2.2 用 lcm 统一「大块」边界

取

$$
\mathrm{lcm}(m,m') = \mathrm{lcm}(4,128) = 128
$$

含义：**每 128 个 token** 构成一个 **对齐超级块**，在这个边界上：

| 在这 128 token 内 | CSA | HCA |
|-------------------|-----|-----|
| 满块 entry 数 | $128/4 = 32$ 条 C4 | $128/128 = 1$ 条 HCA |
| 边界 | 32 条 C4 **刚好填满** | 1 条 HCA **刚好填满** |

任意 **已完成** 的历史长度 $L$ 可写成：

$$
L = 128 \cdot q + r,\quad 0 \le r < 128
$$

- $128q$ token → Classical 池里有 **$32q$ 条 C4** + **$q$ 条 HCA**（一一对应同一批 token 区间）
- 剩余 $r$ token → **进 tail**，**不进** Classical（见下节）

这样 **prefix 共享、落盘、HiSparse 按 entry 搬移** 时，CSA 与 HCA 的「第 $k$ 个 128-token 段」语义一致，不会出现半块只压了一边的情况。

### 2.3 小例子：$L=130$ token

| 区间 | Classical 池 | State 池 |
|------|--------------|----------|
| token $1\ldots128$ | 32×C4 + 1×HCA（已满） | — |
| token $129\ldots130$ | **不写**（未满 4 也未满 128 的 **公共尾**） | **Tail** 2 token + **SWA** 窗口（含最近 128，与上表重叠由 kernel 定义） |

若把 129–130 强行写入 Classical，要么 **伪造半条 HCA**（算法不允许），要么 **破坏 C4 四元组完整性**。

---

## 3. 尾缓冲（tail buffer）的逻辑

### 3.1 流式 decode：来一 token，不能立刻压块

Prefill / decode **逐个 append** token 时，序列长度 $L$ 任意时刻 **不一定** 被 4 或 128 整除。

**Tail buffer** = 当前 **还凑不成一整块** 的 K/V 暂存区：

| 路径 | 何时「满块」 | Tail 最多多长 |
|------|--------------|---------------|
| CSA | $L \bmod 4 = 0$ | 3 token |
| HCA | $L \bmod 128 = 0$ | 127 token |

满块时：**压缩 → 一条 entry append 到 Classical 池 → tail 中对应 token 清空或前移**。

### 3.2 为何 tail 必须在 State 池

| 若在 Classical | 若在 State（实际） |
|----------------|-------------------|
| entry 未定型却要占位 → **无法 immutable 共享** | 明确标记为 **进行中** |
| prefix 命中时要拷贝 **半块垃圾** | 命中后只需 **重算 / 补 tail + SWA**（见 [磁盘 prefix](../../06-推理基础设施/07-V4-磁盘Prefix-Cache.md)） |
| offload 无法判断 **是否完整块** | Classical 里 **全是满块**，HiSparse 可安全搬 **inactive C4** |

Tail 是 **块对齐的工程缓冲**，不是新 attention 类型（详见 [tail buffer 答疑](v4-tail-buffer.md)）。

### 3.3 与 SWA 同池、不同职责

二者都在 State 池，但：

| | **Tail** | **SWA** |
|--|----------|---------|
| 目的 | 等凑满 **压缩块** | 保留 **局部精度** |
| 变化方式 | 长度 $0\ldots m{-}1$ 阶跃 | 窗口 **滑动** |
| 与 Classical 关系 | 满则 **迁入** | **永不**压成 CSA/HCA entry |

---

## 4. 串起来：一次 decode 步发生了什么

设当前已满 Classical 的历史为 $128q$ token，State 里有 tail + SWA。

1. 新 token 到达 → K/V 写入 **tail**（tail 长度 $+1$）；
2. 若 tail 使 CSA 满 4（或 HCA 满 128）→ **触发压缩**，新 entry **append Classical**；
3. **SWA** 窗口同步滑动，丢弃最老 token、纳入最新 token；
4. Attention kernel：**Classical** 上稀疏/密集读压缩 entry + **State** 上读 tail/SWA 精确局部。

**分池** = 第 2 步写完的 **只读历史** 与第 1、3 步的 **可变状态** 永不混排。

---

## 5. 相关阅读

| 文档 | 内容 |
|------|------|
| [v4-kv-layout.md](../../06-推理基础设施/05-V4-KV-Layout.md) | Classical + State 双池定义 |
| [v4-tail-buffer.md](v4-tail-buffer.md) | Tail 生命周期 |
| [v4-swa-sliding-window.md](v4-swa-sliding-window.md) | SWA 为何独立 eviction |
| [v4-hisparse.md](../../06-推理基础设施/06-V4-HiSparse.md) | 为何 offload 主要针对 Classical 里 inactive C4 |
