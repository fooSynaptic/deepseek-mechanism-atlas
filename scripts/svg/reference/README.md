# DeepSeek 图示数学排版参考

与维护约定 [docs/material/meta/svg-diagram-math.md](../../../docs/material/meta/svg-diagram-math.md) 一致。

## 标杆

| 文件 | 说明 |
|------|------|
| [diagrams/mtp-fusion-scheme.svg](../../../diagrams/mtp-fusion-scheme.svg) | MTP 中间 token 融合：顶栏 Eq.21–23、A/B/C phase、框内 LaTeX 记法 |
| [scripts/svg/gen_mtp_fusion_scheme_svg.py](../gen_mtp_fusion_scheme_svg.py) | 生成器（必读模板） |

Engram 基础标杆：[engram-01d-multi-head-hash.svg](../../../docs/material/papers/engram/diagrams/engram-01d-multi-head-hash.svg)

## 辅助库

[svg_math_helpers.py](./svg_math_helpers.py) — 在 Engram 同源 helpers 上扩展：

- `t_supsub` — $h_t^{(k)}$ 上下标
- `t_emb` / `t_rmsnorm` / `t_concat_norms` — V3 MTP 融合记法
- `t_trm` / `t_softmax` / `t_loss` / `t_lambda` / `t_arrow`
- `math_line` / `default_math_styles`

## 新建架构图 checklist

1. 顶栏 `math_line` 与正文 / qa 主公式一致
2. `phase()` 分栏（主路径 → 展开链 → 训练/推理对照）
3. 模块配色与 `gen_mtp_fusion_scheme_svg.py` 常量对齐
4. 框内公式用 `tspan` 片段，不用 `$...$`
5. 改生成器后执行 `python3 scripts/svg/check_svgs.py`
