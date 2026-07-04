---
description: SVG 图示内数学符号排版（tspan 方案，对齐 LaTeX 观感）
globs: papers/**/diagrams/**/*.py,papers/**/diagrams/**/*.svg,**/scripts/svg/gen_*.py,**/diagrams/reference/**
alwaysApply: true
---

# SVG 图示数学排版约定

[论文导读 / qa 中 **Markdown 公式** 用 `$...$`](./markdown-latex-katex.md)。

**SVG 架构图内的公式**不走 KaTeX，而用 **SVG `<tspan>` 排版**，使预览效果接近 LaTeX。

## 标杆实现

| 档位 | 成品 SVG | 生成器 | 适用 |
|------|----------|--------|------|
| **基础** | [`engram-01d-multi-head-hash.svg`](../papers/engram/diagrams/engram-01d-multi-head-hash.svg) | `scripts/svg/engram/gen_engram_01d_svg.py` | 下标变量、顶栏单行公式 |
| **架构 + 公式** | [`mtp-fusion-scheme.svg`](../../../diagrams/mtp-fusion-scheme.svg) | [`gen_mtp_fusion_scheme_svg.py`](../../../scripts/svg/gen_mtp_fusion_scheme_svg.py) | 上下标、算子、分栏数据流、框内公式 |

辅助代码（按仓库选路径，**禁止**生成器内复制粘贴 tspan 片段）：

| 仓库 | 路径 |
|------|------|
| Engram / papers 通用 | [`scripts/svg/engram/reference/svg_math_helpers.py`](../../../scripts/svg/engram/reference/svg_math_helpers.py) |
| DeepSeek / MTP / V3+（**扩展版**） | [`scripts/svg/reference/svg_math_helpers.py`](../../../scripts/svg/reference/svg_math_helpers.py) |

扩展版在基础版上增加：`t_supsub`、`t_emb`、`t_rmsnorm`、`t_trm`、`t_concat_norms`、`t_softmax`、`t_loss`、`t_lambda`、`t_arrow` 等（见下节）。

## 默认方案：Python 生成器 + tspan

| ✅ 做法 | ❌ 禁止 |
|--------|--------|
| 变量斜体：`<tspan font-style="italic">h</tspan>` | 在 SVG 里写 `$h_t^{(k)}$` |
| 下标：`baseline-shift="sub" font-size="0.72em"` | Unicode 下标 `₀₁₂`、raw `α` `φ` `Δ` |
| **上标**：`baseline-shift="super" font-size="0.72em"`（如 $h_t^{(k)}$） | 用 `h^(k)` 纯 ASCII 冒充上标 |
| 算子正体：`RMSNorm(`、`Emb(`、`softmax(`、`OutHead` | 算子也设 `font-style:italic` |
| 希腊字母：数字实体 `&#x3BB;`（λ）、`&#x3C6;`（φ） | 裸写 Unicode `λ` `φ` |
| 数学字体：`Cambria Math, STIX Two Math, Latin Modern Math, serif`（class `.m` / `.fm`） | 全文只用 `system-ui` |
| 中文/英文标签：`system-ui, sans-serif`（`.t` `.lb` `.b` `.an`） | 公式与标签混用同一 font |
| `math_line()` 顶栏 + `parts` 列表拼接 | 手改成品 `.svg` |
| 改图 = 改 `scripts/svg/gen_*_svg.py` + `scripts/svg/check_svgs.py` | 绕过生成器 |

### `tspan` 片段 API

生成器内用 **片段组合** 拼公式，与正文 `$...$` **同一套符号**：

| 辅助函数 | 图内效果 | 正文 LaTeX |
|----------|----------|------------|
| `t_var("x", "t+1")` | $x_{t+1}$ | `$x_{t+1}$` |
| `t_supsub("h", "t", sup="(k)")` | $h_t^{(k)}$ | `$h_t^{(k)}$` |
| `t_supsub("h", "t", sup="'(k)")` | $h_t'^{(k)}$（撇在上标前） | `$h_t'^{(k)}$` |
| `t_emb("t+k")` | $\mathrm{Emb}(x_{t+k})$ | `$\mathrm{Emb}(x_{t+k})$` |
| `t_rmsnorm(inner)` | $\mathrm{RMSNorm}(\cdots)$ | `$\mathrm{RMSNorm}(\cdots)$` |
| `t_concat_norms(h, emb)` | $\mathrm{RMSNorm}(h);\mathrm{RMSNorm}(\mathrm{Emb})$ | 论文 Eq. 拼接 |
| `t_trm("k")` | $\mathrm{TRM}_k$ | `$\mathrm{TRM}_k$` |
| `t_var("M", "k")` | $M_k$ | `$M_k$` |
| `t_softmax(inner)` | $\mathrm{softmax}(\cdots)$ | `$\mathrm{softmax}(\cdots)$` |
| `t_loss("main")` / `t_loss_mtp("k")` | $\mathcal{L}_{\mathrm{main}}$ / $\mathcal{L}_{\mathrm{MTP}}^{(k)}$ | 损失下标 |
| `t_lambda("k")` | $\lambda_k$ | `$\lambda_k$` |
| `t_arrow()` | `→`（实体 `&#x2192;`） | `$\to$` |
| `t_greek("phi", "n,k")` | $\varphi_{n,k}$ | `$\varphi_{n,k}$` |

顶栏主公式示例（与 [`gen_mtp_fusion_scheme_svg.py`](../../../scripts/svg/gen_mtp_fusion_scheme_svg.py) 一致）：

```python
math_line(W // 2, 48, [
 t_supsub("h", "t", sup="'(k)"),
 "<tspan> = </tspan>",
 t_var("M", "k"),
 "<tspan> [ </tspan>",
 t_concat_norms(t_supsub("h", "t", sup="(k-1)"), t_emb("t+k")),
 "<tspan> ]</tspan>",
])
```

框内公式：`_box_math(cx, cy, w, h, fill, stroke, parts)`，节点标签与公式 **同一 `<text>` 内 tspan 拼接**。

## 架构图布局

复杂机制图（训练/推理数据流、因果链）按 **分栏 phase + 横向链 + 侧向虚线** 组织，见 `mtp-fusion-scheme.svg`：

| 元素 | 约定 |
|------|------|
| **标题** | `plain(..., cls="t")` 中文 + 论文锚点（如 `V3 Eq.21-23`） |
| **顶栏公式** | `math_line` 一行，与正文主公式一致 |
| **Phase 虚线框** | `phase(x, y, w, h, "A …")` 分 A/B/C…；A=主路径，B=展开链，C=训练 vs 推理对照 |
| **模块色** | 主网蓝 `#eef4fc` / MTP 绿 `#f0faf0` / 融合橙 `#fef3c7` / token 橙 `#fff8ee` / 头紫 `#ede9fe` |
| **实线箭头** | 主数据流（`arr-b` 蓝 / `arr-g` 绿） |
| **虚线箭头** | 侧向注入（embed、路由，`arr-d`） |
| **链式重复** | 同一行 `Fusion → TRM_k → h_t^{(k)} → x_{t+k+1}`；下行 $k{+}1$，竖线接 $h_t^{(k)}$ |
| **底栏澄清** | class `.st`：`NOT softmax(...) 无依赖并行 …` 否定常见误解 |
| **双栏对照** | 末 phase 左右两框（训练 teacher forcing vs 推理 draft embed） |

样式：`default_math_styles()` 提供 `.m`（12px 公式）、`.fm`（10px 框内公式）、`.an`（蓝色注解）、`.st`（灰色 NOT 行）。

## 符号与正文对齐

图内符号须与 **overview / 论文 / qa** 同一套记法：

| 概念 | 图内写法 | 正文 LaTeX |
|------|----------|------------|
| n-gram（Engram） | `g` + sub `t,n` | $g_{t,n}$ |
| hash | `&#x3C6;` + sub `n,k` | $\varphi_{n,k}$ |
| MTP hidden | `t_supsub("h","t", sup="(k)")` | $h_t^{(k)}$ |
| 中间 token embed | `t_emb("t+k")` | $\mathrm{Emb}(x_{t+k})$ |
| 融合模块 | `t_var("M","k")` | $M_k$ |
| 门控记忆 | `v~` / `V~`（ASCII tilde） | $\tilde{v}$ |

## 备选：TikZ 路径

公式极复杂且环境有 `pdflatex` + `pdf2svg` 时，可用 TikZ 源编译为 SVG（见 Engram `diagrams/reference/build_engram_01d.sh`）。**无 TeX 时回退** Python 生成器。新图优先 tspan + `svg_math_helpers`。

## 与 svg-wiki-diagrams 的关系

- **嵌入**：`<img src="...svg">` → [`svg-wiki-diagrams.mdc`](./svg-diagram-math.md)
- **图内数学**：本 rule
- **交付**：`python3 scripts/svg/gen_*_svg.py && python3 scripts/svg/check_svgs.py`

## 反例

- ❌ SVG 内写 `$h_t^{(k)}=\mathrm{MTPBlock}_k(\cdots)$` 原样显示美元号
- ❌ 用 `h_t^(k)` 或 Unicode 上下标冒充 LaTeX
- ❌ 架构数据流图不用 phase 分栏、框内公式与顶栏公式符号不一致
- ❌ 有 `t_supsub` 等扩展需求却不用 `deepseek-ai/scripts/svg/reference/svg_math_helpers.py`
- ❌ 图内符号与正文 / qa 的 $h_t^{(k)}$、$\mathrm{Emb}(x_{t+k})$ 记法分裂
