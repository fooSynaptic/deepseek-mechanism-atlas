# E3 — GShard×1.5 (closed)

Paper Table 2 / §4.3: DeepSeekMoE 2B **matches** a GShard model with **1.5× expert width** (hence 1.5× expert params and 1.5× activated expert FLOPs). This phase tests that sentence on the same open mix as E1.

**Status:** Closed. Job `e3_gshard_x15` → val BPB **0.7860**; E3-A/B/C all hit.

E1 already ranked DeepSeekMoE ahead of matched-FLOP GShard (Δ = −0.012 BPB). C2 magnitude (≥0.02) missed; E3 still ran because the paper’s “matches ×1.5 compute” claim is independent of that bar.

---

## 1. What is locked

| Item | Value |
| --- | --- |
| Skeleton | Same as E1: 9L / d=1280 / 10 heads / seq 2048 |
| Data | Frozen `data_l2`, GPT-2 50k, val 50M |
| Tokens | **10B** (same D as E1; shorter than 40B / 100B runs) |
| Parallelism | Data parallel (closed `WORLD=20` batch recipe) |
| Parallel | Data parallel only; no EP; α₁ = 0.01 |
| Seed | 1 |
| Batch | ~4M tokens (same recipe as E1) |

**Do not retrain** DeepSeekMoE or matched GShard. Fixed references from E1 LB:

| Arm | Job | val BPB @ 10B |
| --- | --- | ---: |
| GShard (1.0×) | `e1_lb_gshard_w20` | 0.7970 |
| DeepSeekMoE | `e1_lb_dsmoe` | **0.7850** |

---

## 2. Arm under test

**GShard×1.5 (`gshard_x15`).** 16 full SwiGLU experts, Top-2. Each expert intermediate dim = `round_to_64(1.5 × ffn_hidden(d))`.

With `d=1280`, `ffn_hidden=3392`:

| | GShard 1.0× | GShard×1.5 | DeepSeekMoE |
| --- | ---: | ---: | ---: |
| Experts | 16 × h | 16 × 1.5h | 64 × h/4 |
| Expert params / dense FFN | 16.0× | **~24.0×** | 16.0× |
| Activated / dense FFN | 2.0× | **~3.0×** | 2.0× |
| Activated FLOPs vs GShard 1.0× | 1.0× | **~1.5×** | 1.0× |

C1’s “match GShard to 1%” does **not** apply to this arm. Log `expert_over_dense` ≈ 24 and `activated_over_dense` ≈ 3.

---

## 3. Primary question and gates

**Question:** After 10B on this mix, is DeepSeekMoE’s val BPB **close to** GShard×1.5 (paper: comparable), while GShard×1.5 **beats** matched GShard 1.0×?

| Gate | Result |
| --- | --- |
| **E3-A** ×1.5 beats GShard 1.0× | **Hit** — 0.7860 &lt; 0.7970 |
| **E3-B** DSMoE ≈ ×1.5 | **Hit** — \|0.7850 − 0.7860\| = 0.001 ≤ 0.02 |
| **E3-C** No collapse | **Hit** — max_frac end ≈ 0.202 |

Wall actual: **32.5 h** (within the 30–36 h estimate).

Result: `artifacts/e3_archive/`. Full write-up: [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) §E3.


---

## 5. Out of scope

- Extending E1 arms to 40B / 100B
- GShard×1.2
- Dense ×16 upper bound
- Retraining DSMoE at ×1.5 FLOPs
- New tokenizer / Pile train

---

## 6. Pre-registered caption

> At the paper’s 2B skeleton on frozen `data_l2`, after 10B tokens, **GShard with 1.5× expert width (Top-2)** is compared to the existing DeepSeekMoE E1 checkpoint. A hit is: ×1.5 beats matched GShard 1.0×, and DeepSeekMoE val BPB is within 0.02 of ×1.5. Absolute paper Pile numbers are out of scope as a target.
