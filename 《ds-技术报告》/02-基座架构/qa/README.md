# 版本 / MoE 答疑

> [← 版本梗概索引](../../01-总览/02-版本梗概索引.md) · [← 演进总览](../../01-总览/01-版本演进总览.md) · [书中答疑](.)

| 主题 | 来源 | 文件 |
|------|------|------|
| V1：$C=M\cdot D$ vs $C=6ND$ | [演进总览 §3.1 V1](../../01-总览/01-版本演进总览.md#31-deepseek-llm-v1) · [V1 §3](../../04-版本代际/00-V1-LLM.md#3-scaling-laws) | [为何 DeepSeek 用 $C=M\cdot D$ 而非 $C=6ND$？](../../01-总览/qa/v1-scaling-law-c-vs-md.md) |
| MoE centroid vs gate-weight | [DeepSeekMoE §前向公式](../05-DeepSeekMoE.md#forward-formula) | [MoE 路由：gate-weight 还是 expert centroid？](moe-centroid-vs-gate-weight.md) |
| Fine-grained vs GShard 粗专家 | [DeepSeekMoE §优化逻辑 (b)](../05-DeepSeekMoE.md#optimization-logic) | [Fine-grained Expert Segmentation：为何优于 GShard 式粗专家？](moe-fine-grained-segmentation.md) |
| Expert Parallel（EP）与 gather/scatter | [Hash MoE §1.3](../../04-版本代际/06-Hash-MoE-FP4.md#still-inherited) · [DeepSeekMoE §推理 infra](../05-DeepSeekMoE.md#推理-infra-关注点) | [MoE 推理：Expert Parallel与 gather/scatter](moe-expert-parallel-ep.md) |
| Hash MoE 为何只改浅层 | [Hash MoE §1.2](../../04-版本代际/06-Hash-MoE-FP4.md#hash-moe-routing) · [EP 答疑 §4](moe-expert-parallel-ep.md#4-hash-moe-改什么不改什么) | [Hash MoE 为何只改浅层、深层仍用 centroid 路由？](hash-moe-shallow-vs-deep.md) |
| V4 **SWA**（滑动窗口） | [CSA/HCA 一句话](../../04-版本代际/05-CSA-HCA混合压缩注意力.md#一句话) · [KV Layout §State](../../06-推理基础设施/05-V4-KV-Layout.md#state-cache) | [V4 里的 SWA是什么？](../../04-版本代际/qa/v4-swa-sliding-window.md) |
| V4 **Indexer KV** | [CSA/HCA §4](../../04-版本代际/05-CSA-HCA混合压缩注意力.md#v4-mixed-attention) · [Lightning Indexer](../../05-DSA稀疏注意力/04-Lightning-Indexer详解.md) | [V4 里的 Indexer KV 是什么？](../../04-版本代际/qa/v4-indexer-kv.md) |
| V4 **Tail buffer** | [CSA/HCA §4](../../04-版本代际/05-CSA-HCA混合压缩注意力.md#v4-mixed-attention) · [KV Layout §State](../../06-推理基础设施/05-V4-KV-Layout.md#state-cache) | [V4 里的 Tail buffer 是什么？](../../04-版本代际/qa/v4-tail-buffer.md) |
| V4 **分池 / 块对齐 / 尾缓冲** | [KV Layout §为何不够](../../06-推理基础设施/05-V4-KV-Layout.md#为什么-v3v32-的单一-layout-不够) | [V4 为何要分池？块对齐与尾缓冲怎么配合？](../../06-推理基础设施/qa/v4-kv-dual-pool-alignment.md) |
| H2D / D2H、PCIe prefetch | [ESS 论文梗概 §Fig.6&7](../../06-推理基础设施/02-ESS论文梗概.md#figure-67-overlap-策略对比) | [H2D / D2H 是什么？](../../06-推理基础设施/qa/h2d-d2h-pcie-transfer.md) |
| FP8 partial sum 漂移 | [FP8 专文 §为何需要动态量化](../06-V3-FP8动态量化.md#为何需要动态量化) | [FP8 动态量化里的 partial sum 漂移](fp8-partial-sum-drift.md) |
| 投机解码 lossless / 接受路径 | [投机解码专文 §1.3](../../06-推理基础设施/04-DSpark投机解码.md#13-净收益lossless-与最坏情况) | [投机解码：为何接受率是 $\min$、修正分布是 $\max$](../../06-推理基础设施/qa/spec-decode-rejection-sampling.md) |
| Compute-Bound vs Memory-Bound；DFlash / Eagle | [专文 §1.1](../../06-推理基础设施/04-DSpark投机解码.md#11-瓶颈每-token-搬一遍大模型权重) · [§4](../../06-推理基础设施/04-DSpark投机解码.md#4-草稿范式总览eagle3--dflash--mtp--dspark) · [酱紫君 §Speculative Decoding](../../08-外部解读/03-酱紫君DSpark阅读笔记.md#speculative-decoding) | [投机解码：Compute-Bound vs Memory-Bound — DFlash / Eagle 如何对应？](../../06-推理基础设施/qa/spec-decode-compute-vs-memory-bound.md) |
| **SM**（Streaming Multiprocessor） | [专文 §1.1](../../06-推理基础设施/04-DSpark投机解码.md#11-瓶颈每-token-搬一遍大模型权重) · [Compute vs Memory §2](../../06-推理基础设施/qa/spec-decode-compute-vs-memory-bound.md#2-为何-decode-默认是-memory-bound) | [名词解释：SM](../../06-推理基础设施/qa/gpu-sm-term.md) |
| **MTP 中间 token 融合** | [专文 §2](../../06-推理基础设施/04-DSpark投机解码.md#2-deepseek-路线mtpv3--v4) · [§1.1 链深度](../../06-推理基础设施/qa/mtp-fusion-scheme.md#11-你的理解对在哪里错在哪里) | [MTP 融合 scheme](../../06-推理基础设施/qa/mtp-fusion-scheme.md) · [MTP draft 链深度图](../../06-推理基础设施/figures/mtp-draft-chain-depth.svg) |
| **Birkhoff 多面体** | [mHC §3.2](../../04-版本代际/04-mHC流形约束超连接.md#32-birkhoff-多面体) · [§3 双随机流形](../../04-版本代际/04-mHC流形约束超连接.md#3-mhc-核心双随机流形约束) | [名词解释：Birkhoff 多面体](../../04-版本代际/qa/mhc-birkhoff-polytope.md) |
| MMA / WGMMA 名词 | [FP8 专文 一句话](../06-V3-FP8动态量化.md) · [partial sum 答疑](fp8-partial-sum-drift.md) | [名词解释：MMA](fp8-mma-term.md) |
