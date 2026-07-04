# L1 HBM / L2 DRAM / L3 CXL.mem：三级存储区别

[← 返回 §CXL 三级存储](../engram-series-overview.md#体系结构三级存储分层) · [硬件拓扑 02b](../diagrams/engram-02b-cxl-hardware-topology.svg) · [访问时序 02c](../diagrams/engram-02c-cxl-cache-access.svg) · [prefetch window](./cxl-prefetch-window-cpu-gpu.md) · [CXL vs RDMA](./cxl-vs-rdma-communication-pattern.md) · [答疑目录](./README.md)

---

## 问题

[CXL 硬件拓扑图](../diagrams/engram-02b-cxl-hardware-topology.svg) 里出现 **L1 HBM、L2 Local DRAM、L3 CXL Pool**——三者物理上是什么？在 Engram 部署里各放什么、差在哪？

---

## 结论（先答）

| | **HBM（L1）** | **Host DRAM（L2）** | **CXL.mem（L3）** |
|--|---------------|---------------------|-------------------|
| **物理位置** | 焊在 **GPU 封装内** | **CPU 插槽旁** 主板内存条 | **CXL 扩展内存卡**，经 Switch 挂到 PCIe/CXL  fabric |
| **谁直接访存** | **GPU SM**（最高带宽） | **CPU**；GPU 经 PCIe 间接读 | **CPU / GPU** 经 CXL 协议 **load/store**（像内存，不像网卡消息） |
| **典型容量** | 数十 GB / 卡 | TiB 级 / 节点 | 论文 **256 GB/卡**；多机池 **4 TB / 8 机** |
| **延迟** | 最低（GPU 片上） | 本地 DRAM 级 | 论文：**≈ 本地 DRAM**（远好于小包 RDMA） |
| **带宽特点** | 极高（TB/s 级），**容量瓶颈** | 高，**按节点私有** | 交换机聚合 **512 GB/s**；**多机共享一份表** |
| **Engram 放什么** | MoE/Attn **权重**、hidden、**gate/conv 小参数**、prefetch **staging** | **热 embedding 缓存**、CPU `offsets[]`、bounce buffer、CXL 的 **mmap 窗口** | **完整 Engram embedding 大表**（只读、DAX mmap） |
| **大表能否全放下** | ❌ 装不下百亿参数表 | ⚠️ 单机可放一部分；**多机各拷一份贵** | ✅ **机架级共享一份** |

**一句话**：**HBM = GPU 算力旁边的小而快内存**；**DRAM = CPU 旁边的大内存（可当中间缓存）**；**CXL.mem = 用内存语义挂在 fabric 上的超大扩展池，专扛 Engram 只读大表 + 多机共享**。

---

## 1. 三种介质分别是什么

### HBM（High Bandwidth Memory）

- **是什么**：与 GPU die **2.5D/3D 封装**的专用 DRAM 栈（如 H100 上的 HBM2e/HBM3）。
- **特点**：带宽极高、延迟最低，但 **每卡只有几十 GB～百余 GB**，且 **只有 GPU 能高效用**。
- **类比**：GPU 的「工作台台面」——算子权重、激活、临时 buffer 必须在这儿才能跑满算力。

### Host DRAM（本地 DDR）

- **是什么**：服务器主板上 **DDR4/DDR5 内存条**，CPU 通过内存控制器访问。
- **特点**：容量 **比 HBM 大一个数量级以上**（单节点可达 TB），延迟低，但 **GPU 访问要经过 PCIe**（除非 pinned + DMA）。
- **Engram 语境**：奠基论文把大表 **offload 到 Host DRAM**，异步 prefetch 进 HBM，吞吐损失 **<3%**；CXL 论文里 L2 还可做 **Zipf 热 n-gram 缓存**，命中则不必每次打到 L3。

### CXL.mem（CXL 内存扩展）

- **是什么**：符合 **CXL.mem** 协议的 **扩展内存设备**（内存语义设备），经 **CXL Adapter + Switch** 接到 CPU/GPU 同一 fabric（论文用 XConn XC50256）。
- **与「普通 DRAM」的区别**：
  - **不在** 主板 DIMM 槽里，而是 **扩展卡 / 池化设备**；
  - 多主机可通过 Switch **映射到各自地址空间**，**共享物理一份**数据（Engram 表只读，无需 cache coherence）；
  - 软件用 **DAX + mmap**：`cxl_base + offset[i]` **按 cache line 细粒度 load/store**，适合 Engram **5 KB×多段离散小读**。
- **与 RDMA 池的区别**：RDMA 是 **消息式** 远端读写，小包效率差；CXL 是 **字节寻址内存**，延迟接近 DRAM（见 CXL 论文 Figure 3/5/6）。

---

## 2. Engram 三级分层（L1 / L2 / L3）

CXL 论文 Figure 4 把部署抽象为三层（与 [02b 拓扑图](../diagrams/engram-02b-cxl-hardware-topology.svg) 一致）：

```
L3 CXL Pool     完整 embedding 表（只读，多机一份）
      | prefetch / memcpy
L2 Host DRAM    热缓存、CPU 算 offsets、可选中转
      | DMA / P2P
L1 GPU HBM      staging e_t -> W_K,W_V, gate, conv -> 写回 H
```

| 步骤 | 典型位置 | 说明 |
|------|----------|------|
| hash / offsets | **CPU + L2** | 只依赖 input ids（见 [Step 3 图 CPU 区](../diagrams/engram-01d-multi-head-hash.svg)） |
| 读表 | **L3**（未命中热缓存时） | 离散小读；可 overlap 前层 GPU 计算 |
| 热命中 | **L2 或 L1 缓存** | Zipf：高频 n-gram 不必每次访问 L3 |
| gate / conv / Attn | **L1 HBM** | 需要 $h_t$，必须在 GPU |

数据通路（论文 Listing 1/2）：

- **CXL → CPU**：OpenMP 并行 `memcpy`（延迟 ≈ DRAM）
- **CXL → GPU（推荐）**：`cxl2vram_copy` P2P，离散 embedding 直进 **HBM staging**

详见 [02c 访问时序](../diagrams/engram-02c-cxl-cache-access.svg)。

---

## 3. 为什么 Engram 需要三层，而不是「全放 HBM」

1. **表太大**：百亿参数级 embedding **放不进 HBM**；全复制到每机 DRAM **贵且浪费**。
2. **访问又小又碎**：每 token 每层约 **5 KB**、多段随机读——适合 **内存语义细粒度读**，不适合 RDMA 小包风暴。
3. **索引可提前算**：与 GPU 算力 **overlap prefetch**（~56 μs 窗口 / 层，见 overview §②）。
4. **多机推理**：**L3 共享一份表** + L2/L1 热缓存，比每节点 DRAM 全副本更省。

---

## 4. 和「奠基论文只用 DRAM offload」的关系

| 阶段 | 存储 |
|------|------|
| Engram ①（2601.07372） | 大表 offload 到 **Host DRAM**，PCIe prefetch 到 HBM |
| CXL Pooling ②（2603.10087） | 表升级到 **L3 CXL 共享池**；L2 热缓存 + L1 staging 逻辑 **不变** |

CXL 论文结论：端到端吞吐 **CXL ≈ DRAM**（如 Qwen3-4B **1.2%** 损失），成本上多机大表更优。

---

## 5. 速查：该把什么放在哪

| 数据 | 推荐层级 |
|------|----------|
| Transformer / MoE 权重 | **L1 HBM** |
| 当前 hidden、KV（若在本机） | **L1 HBM** |
| Engram gate / ShortConv 参数 | **L1 HBM**（小） |
| prefetch 到的 $e_t$ / embedding 行 | **L1 staging** |
| 高频 embedding 缓存 | **L2**（可选 **L1** 更小缓存） |
| 完整 Engram 表 | **L3 CXL**（或单机时 **L2 DRAM**） |
| hash `offsets[]` 计算 | **CPU**（不占 HBM 算力） |

---

## 参考

- [engram-series-overview.md §② 三级存储](../engram-series-overview.md#体系结构三级存储分层)
- [engram-02b-cxl-hardware-topology.svg](../diagrams/engram-02b-cxl-hardware-topology.svg)
- [engram-02c-cxl-cache-access.svg](../diagrams/engram-02c-cxl-cache-access.svg)
- [engram-01d-multi-head-hash.svg](../diagrams/engram-01d-multi-head-hash.svg) — CPU/GPU 分界
- CXL 论文 arXiv:2603.10087 · [cxl-2603.10087.pdf](../src/cxl-2603.10087.pdf)
