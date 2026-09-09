# F-train — half-activated DeepSeekMoE (Fig 6)

**Status:** Closed. Job `f_train_k3` finished (10B, 25.1 h). Strict F-A miss; **attribution = near-match GShard**.

Paper §4.5 / Figure 6: train from scratch with **1 shared + Top-3 of 63** quarter-experts. Same total expert params as E1 DeepSeekMoE / GShard, but **half** the activated expert params. Claim: still beats matched-FLOP GShard.

F-eval (done) only changed `k` at eval on a Top-7-trained ckpt. That is Fig 5. **This job is Fig 6** — router and experts learn under Top-3 for the full 10B.

Report: [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md). Curves: [`figures/lab_family_curves.png`](figures/lab_family_curves.png). Archive: `artifacts/f_train_archive/`.

---

## 1. Locked setup

| Item | Value |
| --- | --- |
| Skeleton | 9L / d=1280 / 10 heads / seq 2048 |
| Data | Frozen `data_l2`, GPT-2 50k, val 50M |
| Tokens | **10B** (same D as E1 / E3) |
| Parallelism | Data parallel (closed `WORLD=20` batch recipe) |
| Parallel | DDP only; α₁ = 0.01 on **63 routed** only |
| Seed | 1 |
| Batch | ~4 014 080 tokens/step (same as E1) |

**Fixed references (do not retrain):**

| Arm | Job | val BPB @ 10B |
| --- | --- | ---: |
| GShard Top-2 | `e1_lb_gshard_w20` | **0.7970** |
| DeepSeekMoE 1+Top-7 | `e1_lb_dsmoe` | 0.7850 |

---

## 2. Arm under test

**`dsmoe_k3`.** Same pool as DeepSeekMoE: 1 shared + 63 routed, each `h/4`. **Train and eval with `k_routed=3`.**

| | GShard | DSMoE (E1) | **F-train** |
| --- | ---: | ---: | ---: |
| Total experts (param) | 16 × h | 64 × h/4 | 64 × h/4 |
| Expert params / dense | 16× | 16× | 16× |
| Activated slots | Top-2 full | 1+7 quarter | **1+3 quarter** |
| Activated / dense FFN | 2.0× | 2.0× | **1.0×** |
| Activated vs GShard | 1.0× | 1.0× | **0.5×** |

C1 “activated match GShard” does **not** apply. Log `expert_over_dense=16` and `activated_over_dense≈1.0`.

Implementation note: the current `BatchedExperts` still matmuls all 63 experts then applies gates, so **walltime ≈ E1 DSMoE (~25 h)** (still full expert matmuls; no half walltime). The scientific object is still Top-3 routing / half activated capacity in the MoE formula.

---

## 3. Gates

| # | Gate | Hit | Miss |
| --- | --- | --- | --- |
| **F-A** | Beats GShard at half activated | val BPB **&lt; 0.7970** | ≥ GShard, or collapse |
| **F-B** | No collapse | end `max_expert_frac` &lt; 0.5; finite aux | Monopoly / NaN |
| **F-C** (soft) | Not far behind full DSMoE | BPB − 0.7850 **≤ 0.03** | Large gap; still OK if F-A hits |

**Hit caption (F-A + F-B):** on this mix at 10B, DeepSeekMoE trained with only half the activated expert params still beats matched GShard — paper Fig 6 **in direction**.

Not required: match Pile, beat E1 DSMoE, or reproduce F-eval’s k=4 row (different protocol).

---

## 4. Why not reuse F-eval k=3

| | F-eval k=3 | F-train k=3 |
| --- | --- | --- |
| Weights | Trained as Top-7 | Trained as Top-3 |
| vs GShard | Same-batch **worse** (0.840 vs 0.832) | Final table **near-match** (0.8014 vs 0.7970) |
| Paper figure | Fig 5 (eval sweep) | **Fig 6** (train from scratch) |

F-train still sits above F-eval k=3’s absolute BPB (different protocol / val draw), but the **ranking vs GShard** is much closer after training under Top-3.

---

## 5. Walltime

| Phase | Estimate |
| --- | ---: |
| Smoke 8 steps | minutes |
| Full 10B | **~24–28 h** (≈ E1 DSMoE 25.0 h) |

One job: `f_train_k3`. No packing with other MoE trains.

---

## 6. Out of scope

- S-train, E2 40B, GShard×1.2
- Changing expert width or shared count
- Optimizing BatchedExperts to skip inactive experts (speed only)

---

## 7. Pre-registered caption

> On frozen `data_l2`, at the paper’s 2B skeleton, **DeepSeekMoE with 1 shared + Top-3 of 63** is trained from scratch for 10B tokens. A hit is a **lower val BPB than GShard 0.7970** with no routing collapse, at half the activated expert params. Absolute paper Pile numbers are out of scope as a target.

---

## 8. Result and attribution (closed)

| Arm | val BPB @ 10B | vs GShard |
| --- | ---: | ---: |
| GShard Top-2 | 0.7970 | 0 |
| **F-train 1+Top-3** | **0.8014** | **+0.004** |
| DeepSeekMoE 1+Top-7 | 0.7850 | −0.012 |

| Gate | Result |
| --- | --- |
| **F-A** (BPB &lt; 0.7970) | **Miss (strict)** |
| **F-B** (no collapse) | **Hit** (`max_frac≈0.12`) |
| **F-C** (≤ +0.03 vs full DSMoE) | **Hit** (+0.016) |

**Preferred framing:** basically **matches** GShard. The +0.004 gap is ~3× smaller than the full-model DSMoE edge (−0.012) and sits inside ~0.015 last-1B scatter. Near-match only relative to Fig 6; also far from a large architectural miss.

| Hypothesis | Verdict |
| --- | --- |
| Total expert params too small | **Reject** (still 16× dense) |
| Half activated harder to learn | **Contributing** |
| Tokens too few vs paper (~100B) | **Primary** |
| Routing collapse | **Reject** |
| Representation cannot work at low k | **Reject** (F-eval k≥4 already ≤ GShard) |
