# CXL vs RDMA：Engram 的两种「远程内存」通信 pattern

[← 返回 §RDMA 对比](../../../../../07-Engram/02-Engram系列导读.md#与-rdma-池的访问逻辑对比) · [02c 时序图（已标注）](../diagrams/engram-02c-cxl-cache-access.svg) · [为何选 CXL 非 RDMA](cxl-why-cxl-not-rdma.md) · [prefetch window](cxl-prefetch-window-cpu-gpu.md) · [答疑目录](README.md)

---

## 问题

[02c 图](../diagrams/engram-02c-cxl-cache-access.svg) 右下角 **communication pattern** 里 **CXL** 与 **RDMA** 各指什么通信方式？和主流程 Step 2 的箭头如何对应？

---

## 结论（先答）

| | **CXL pattern**（Engram 用） | **RDMA pattern**（KV 池 / Mooncake 类） |
|--|------------------------------|----------------------------------------|
| **语义** | **内存 load/store**（像读本地 DRAM） | **消息 get/put**（像网络 RPC） |
| **寻址** | `mmap(cxl_base)` + `offset[i]` 字节地址 | 远端对象句柄 + 显式请求 |
| **粒度** | **cache-line / 数百字节** 随机读 | 适合 **MB–GB 大块** |
| **典型路径** | L3 CXL → **PCIe P2P** → GPU staging（或经 CPU memcpy） | GPU → host bounce → **NIC RDMA** → 远端 DRAM |
| **Engram 5KB×多段离散读** | 延迟 **≈ DRAM**，可塞进 **~56 μs** 窗口 | 小包效率差，延迟 **远高于 DRAM** |

**02c 主流程画的是 CXL**；灰色 **RDMA pool** 面板是 **对比参照**（同一需求若走 RDMA 为何不行）。

→ 时间窗 + 56 μs 如何推出这一结论：[cxl-why-cxl-not-rdma.md](cxl-why-cxl-not-rdma.md)

---

## 1. CXL pattern（图里橙色 / 黄框）

### 是什么

**CXL.mem** 把扩展内存挂进 CPU/GPU 的 **统一地址空间**。进程 `mmap` 后，读 embedding 就是 ordinary **load/store**：

```
cxl_ptr = mmap(DAX device)
row = load(cxl_ptr + offset[i])   // 离散 320B 行
```

### 在 02c 上标在哪

| 图元素 | pattern |
|--------|---------|
| 黄框 **L3 CXL Pool** | `mmap: cxl_base+offset[i]` · **CXL.mem: load/store** |
| **橙色虚线** path B | **CXL load/store + PCIe P2P**（推荐） |
| 绿框 path A | **CXL→CPU** OpenMP memcpy，再进 GPU |
| 蓝框 **L1 staging** | `cxl2vram_copy` 终点 |
| 右侧橙条 **CXL: fetch @ layer k** | 与 GPU 算 0..k-1 **并行** 的 CXL 读 |

**不是**发「请给我第 i 行」的网络消息，而是 **CPU/GPU DMA 直接按地址读**。

---

## 2. RDMA pattern（图里灰色面板）

### 是什么

**RDMA 内存池**（如 Mooncake 式 KV offload）：GPU/CPU 通过 **NIC** 对远端内存发 **get/put**：

```
put(remote_buf, local_chunk)   // 消息式
get(local_buf, remote_handle)  // 需 NIC 参与、常经 bounce buffer
```

面向 **大块、连续** 传输（整段 KV cache）；对 Engram 这种 **每 token 每层 ~5 KB、16 段离散 320B** 访问：

- 大量 **小包** → 带宽利用率低（论文：64B 级可跌至峰值 **~25%**）
- 软件栈 + NIC 延迟 → **远高于本地 DRAM**，**填不进 56 μs prefetch 窗**

### 在 02c 上标在哪

| 图元素 | pattern |
|--------|---------|
| 右下灰框 **RDMA pool (KV-style)** | `get/put` · GPU→bounce→NIC→remote |
| 小示意图 GPU—NIC—remote | **不走** 主流程 Step 2 |
| 文 **many 320B pkts \| too slow here** | 与 Engram 访问形态不匹配 |

主流程 **没有** RDMA 箭头——故意表示 **Engram 不选这条 pattern**。

---

## 3. 为何 Engram 要标清两者（论文动机）

1. **延迟预算**：Layer 2 的 prefetch 窗 ≈ **56 μs** → 需要 **近 DRAM** 的 CXL，RDMA 不够。
2. **访问形态**：离散小行读 → **load/store** 自然；RDMA **消息** 开销过大。
3. **多机共享**：CXL Switch **硬件映射** 同一 L3 池；RDMA 也能共享，但 **语义不同、延迟曲线不同**。

奠基论文在 **Host DRAM** offload 已验证 prefetch；CXL 论文把 L3 换成 **共享 CXL 池**，**通信 pattern 从「本地 DRAM load/store」换成「CXL load/store」**，而不是换成 RDMA。

---

## 4. 与三级存储的对应

见 [cxl-l1-l2-l3-memory-tiers.md](cxl-l1-l2-l3-memory-tiers.md)：

- **CXL pattern** 发生在 **L3 → L1**（Step 2）
- **RDMA pattern** 是另一种「远端内存」产品形态，**不落在** Engram 02c 主路径上

---

## 参考

- [engram-series-overview.md §与 RDMA 池对比](../../../../../07-Engram/02-Engram系列导读.md#与-rdma-池的访问逻辑对比)
- [engram-02c-cxl-cache-access.svg](../diagrams/engram-02c-cxl-cache-access.svg)
- CXL 论文 Figure 2a vs 2b · arXiv:2603.10087
