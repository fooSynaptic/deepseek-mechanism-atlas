# DeepSeek 技术报告与外部解读

> [← 中文导读](../README.md) · [← 仓库首页（EN）](../../README.md) · [版本梗概](../versions/README.md) · [演进总览](./deepseek-version-lineage-20260625.md) · [Raschka V3→V3.2 解读](./raschka-technical-deepseek-v3-v32-highlights.md)

本目录存放 **官方技术报告摘要** 与 **第三方深度解读**（标注来源，便于与本地梗概对照）。

| 文档 | 类型 | 说明 |
|------|------|------|
| [deepseek-v1-to-v3-lineage.md](./deepseek-v1-to-v3-lineage.md) | 本地总览 | **V1 → V2 → V3** 前代演进 |
| [v1.md](../versions/v1.md) | **精读** | DeepSeek-LLM V1 完整中文译文（2401.02954；Figure 2–5 / Table 3–4） |
| [deepseek-version-lineage-20260625.md](./deepseek-version-lineage-20260625.md) | 本地总览 | 全系列 V1→V4 算法线 + infra 线 |
| [deepseek-algorithm-line.md](./deepseek-algorithm-line.md) | **算法线导读** | MLA → DSA → CSA/HCA + mHC 专题 hub |
| [deepseek-infra-line.md](./deepseek-infra-line.md) | **基础设施线导读** | MLA KV → 异构 Cache → Index Share → ESS → V4 HiSparse |
| [deepseek-moe-line.md](./deepseek-moe-line.md) | **MoE 线导读** | 稠密 FFN → DeepSeekMoE → aux-loss-free → Hash MoE |
| [raschka-technical-deepseek-v3-v32-highlights.md](./raschka-technical-deepseek-v3-v32-highlights.md) | **梗概** | Raschka 一文要点速读 |
| [raschka-technical-deepseek-v3-v32.md](./raschka-technical-deepseek-v3-v32.md) | **全文解析** | 分章整理 + 关键表格嵌入 |
| [zhihu-jiangzijun-dspark-highlights-20260627.md](./zhihu-jiangzijun-dspark-highlights-20260627.md) | **外部解读** | 酱紫君（GalAster）：DSpark、半自回归、验证截断、MTP、draft 训练 |
| [dspark-speculative-decoding.md](../versions/dspark-speculative-decoding.md) | **投机解码全集** | MTP、外挂 draft 自测、DSpark、MTP-1（**唯一入口**） |
| [spec-decode-draft-acceleration-20260604.md](./spec-decode-draft-acceleration-20260604.md) | 重定向 | 已并入上表专文 §3 |
| [deepseek-doc-series-audit-20260627.md](./deepseek-doc-series-audit-20260627.md) | **结构审查** | 双向引用、章节导航、概念/SVG 复用审计 |

**CI 门禁**：`bash scripts/doc_series_gate.sh`（`check_svgs` + `build_book` + FP8 导航 spot-check）

**外部原文**：[A Technical Tour of the DeepSeek Models from V3 to V3.2](https://magazine.sebastianraschka.com/p/technical-deepseek)（Sebastian Raschka，2025-12-03，更新 2026-01-01）
