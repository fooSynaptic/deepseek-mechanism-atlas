# S-train — no shared expert from scratch

**Status:** Closed. Job `s_train_noshare` (`dsmoe_s0`) finished (10B, 25.7 h). **S-A hit (soft)**; S-eval does **not** substitute.

S-eval (done) only drops the shared expert **at eval** on a checkpoint trained **with** a shared expert.

**S-train** asks: if the model **never** has a dedicated shared expert, can Top-8 among 64 routed quarter-experts still match DeepSeekMoE (1+Top-7)?

Report: [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md). Archive: `artifacts/s_train_archive/`. Reproduce: [`REPRODUCE.md`](REPRODUCE.md).

---

## 1. Locked setup

| Item | Value |
| --- | --- |
| Skeleton | 9L / d=1280 / 10 heads / seq 2048 |
| Data | Frozen `data_l2`, GPT-2 50k, val 50M |
| Tokens | **10B** |
| Parallelism | Data parallel (closed `WORLD=20` batch recipe) |
| Parallel | DDP only; α₁ = 0.01 on **64 routed** |
| Seed | 1 |
| Batch | ~4 014 080 tokens/step |

**Fixed references:**

| Arm | Job | val BPB @ 10B |
| --- | --- | ---: |
| GShard Top-2 | `e1_lb_gshard_w20` | 0.7970 |
| DeepSeekMoE 1+Top-7 | `e1_lb_dsmoe` | **0.7850** |

---

## 2. Arm under test

**`dsmoe_s0`.** 64 quarter-size experts, **all routed**, **n_shared=0**, **Top-8**. Same total + activated params as DeepSeekMoE 1+Top-7 (16× / 2.0× dense).

---

## 3. Gates (pre-registered)

| # | Gate | Hit | Miss |
| --- | --- | --- | --- |
| **S-A** | Shared isolation helps from scratch | S-train ≥ 0.7850 + 0.008 | ≤ DSMoE + 0.004 |
| **S-B** | No collapse | `max_expert_frac` &lt; 0.5 | Monopoly / NaN |
| **S-C** (soft) | vs GShard | Report Δ vs 0.7970 | — |

---

## 4. Result (closed)

| Arm | val BPB | vs DSMoE | vs GShard |
| --- | ---: | ---: | ---: |
| DeepSeekMoE 1+Top-7 | 0.7850 | 0 | −0.012 |
| **S-train 0+Top-8** | **0.7933** | **+0.0083** | **−0.0037** |
| GShard Top-2 | 0.7970 | +0.012 | 0 |

| Gate | Result |
| --- | --- |
| **S-A** | **Hit** (+0.0083 ≥ 0.008) |
| **S-B** | **Hit** (`max_frac≈0.26`) |
| **S-C** | Slightly better than GShard |

**Caption:** shared isolation helps from scratch, but softly. Surgery S-eval (+0.135) remains a different, much larger effect.

---

## 5. Reproduce

```bash
python3 scripts/check_arch.py
```

Config: `configs/s_train.json`. Arch: `dsmoe_s0` in `src/model.py` / `src/train.py`. Training orchestration is not published; see archived metrics under `artifacts/s_train_archive/`.

---

## 6. Pre-registered caption

> On frozen `data_l2`, at the paper’s 2B skeleton, **DeepSeekMoE-style experts with 0 shared + Top-8 of 64** are trained from scratch for 10B tokens. The question is whether val BPB is clearly worse than DeepSeekMoE **1 shared + Top-7** (0.7850) under matched total and activated params. Absolute paper Pile numbers are out of scope as a target.
