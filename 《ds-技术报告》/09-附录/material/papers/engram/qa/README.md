> [← 返回 Engram 系列导读](../../../../../07-Engram/02-Engram系列导读.md) · [答疑目录](README.md) · [← 中文导读](../../../../../00-前言/02-中文导读.md) · [← 仓库首页（EN）](https://github.com/fooSynaptic/deepseek-tech-notes)

# Engram 系列 · 答疑

主文档公式/推导类细节放本目录；`engram-series-overview.md` 只保留一行跳转。

| 主题 | 来源章节 | 答疑 |
|------|----------|------|
| Step 6 门控依据 / 记忆依赖过滤 | [§Step 6](../../../../../07-Engram/02-Engram系列导读.md#step-6-上下文门控) | [Step 6 上下文门控：依据与「记忆依赖过滤」](step6-context-gating-rationale.md) |
| HBM / DRAM / CXL.mem（L1–L3） | [§CXL 三级存储](../../../../../07-Engram/02-Engram系列导读.md#体系结构三级存储分层) | [L1 HBM / L2 DRAM / L3 CXL.mem：三级存储区别](cxl-l1-l2-l3-memory-tiers.md) |
| prefetch window / CPU vs GPU | [§缓存访问逻辑](../../../../../07-Engram/02-Engram系列导读.md#缓存访问逻辑一次-decode-的完整路径) | [Prefetch window：不是「CPU 比 GPU 强」，而是 CPU 先点火、GPU 腾出时间窗](cxl-prefetch-window-cpu-gpu.md) |
| CXL vs RDMA 通信 pattern | [§RDMA 对比](../../../../../07-Engram/02-Engram系列导读.md#与-rdma-池的访问逻辑对比) · [02c 图](../diagrams/engram-02c-cxl-cache-access.svg) | [CXL vs RDMA：Engram 的两种「远程内存」通信 pattern](cxl-vs-rdma-communication-pattern.md) |
| ③ Engram-Nine 热/冷 flip | [§③ 核心发现](../../../../../07-Engram/02-Engram系列导读.md#核心发现反直觉) · [论文截图](../assets/figures/nine-2601.16531/) | [DeepSeek Engram 系列导读§③](../../../../../07-Engram/02-Engram系列导读.md#-无冲突热层实验-engram-ninearxiv260116531) |
| 为何选 CXL 而非 RDMA | [§Step 1 时间窗](../../../../../07-Engram/02-Engram系列导读.md#step-1-异步-prefetch-触发) | [为何选 CXL 而非 RDMA？](cxl-why-cxl-not-rdma.md) |
| Step 7 感受野扩充常数 | [§Step 7](../../../../../07-Engram/02-Engram系列导读.md#step-7-短卷积扩感受野) | [Step 7 短卷积：感受野扩充常数](step7-short-conv-receptive-field.md) |
| RF 1→10 对训练/推理的影响 | [§Step 7](../../../../../07-Engram/02-Engram系列导读.md#step-7-短卷积扩感受野) | [Step 7 感受野 1→10：训练与推理差异](step7-rf10-train-infer-impact.md) |
| 记忆过滤在哪一步？ | [§Step 1–8 总表](../../../../../07-Engram/02-Engram系列导读.md#阶段-a确定性检索o1-查表) | [01c 前向图](../diagrams/engram-01c-forward-dataflow.svg) + [Step 6 门控说明](step6-context-gating-rationale.md) |
