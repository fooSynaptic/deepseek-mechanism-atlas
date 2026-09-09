# DeepSeekMoE vs GShard — Experiment Design

## Aim

The paper’s early MoE claim (*DeepSeekMoE*, Dai et al., [arXiv:2401.06066](https://arxiv.org/abs/2401.06066) §4–§5; ~2B total / ~0.3B activated) is that two **layout** changes beat matched-FLOP GShard: **fine-grained segmentation** and **shared-expert isolation** (Figure 2), with total expert params and activated FLOPs locked.

This work seeks to reproduce those paper claims on a convenient open mix ([`data_l2`](#5-data)), at small model scale, with **10B** tokens, GPT-2 50k, and val BPB.

Product DeepSeek-V2/V3 training (236B / 671B) and starting from a pretrained checkpoint are **out of scope**. Absolute paper Pile CE is out of scope as a target.

Schematic: [`figures/dsmoe_vs_gshard.svg`](figures/dsmoe_vs_gshard.svg).

**Parallelism (locked):** data parallel only — full MoE replica per rank; no expert parallel; no token dropping. Closed 10B recipe: `WORLD=20` → ~4M-token global batch (changing `world_size` changes the batch). CPU / single-GPU checks: [`REPRODUCE.md`](REPRODUCE.md).

---

## Status and document map

**Program:** closed at 10B (E1 + LB fix + §4.5 F/S-eval + E3 + F-train + S-train). Optional E2 (40B) incomplete.

| Hub | Open |
| --- | --- |
| Results narrative | [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) |
| Local reproduce (CPU / single-GPU scope) | [`REPRODUCE.md`](REPRODUCE.md) |
| Visual scorecard | [`figures/lab_scorecard.svg`](figures/lab_scorecard.svg) |

| Phase | Status | Design / protocol | Config | Artifacts |
| --- | --- | --- | --- | --- |
| E1 ranking (+ LB fix) | Closed | this doc (§2–§4, §6) | [`configs/e1_lb.json`](../configs/e1_lb.json) | [`artifacts/e1_lb_archive/`](../artifacts/e1_lb_archive/) |
| E1 first run (collapsed) | Invalid | — | [`configs/e1.json`](../configs/e1.json) | [`artifacts/e1_archive/`](../artifacts/e1_archive/) |
| E2 40B (optional) | Incomplete | this doc §6 | [`configs/e2.json`](../configs/e2.json) | [`artifacts/e2_archive/`](../artifacts/e2_archive/) |
| E3 GShard×1.5 | Closed | [`E3_DESIGN.md`](E3_DESIGN.md) | [`configs/e3.json`](../configs/e3.json) | [`artifacts/e3_archive/`](../artifacts/e3_archive/) |
| **F** / **S** eval (§4.5) | Closed | [`ABLATION_ARMS_DESIGN.md`](ABLATION_ARMS_DESIGN.md) | — (eval on E1 ckpt) | [`artifacts/fs_eval_archive/`](../artifacts/fs_eval_archive/) |
| **F**-train (1+Top-3) | Closed | [`F_TRAIN_DESIGN.md`](F_TRAIN_DESIGN.md) | [`configs/f_train.json`](../configs/f_train.json) | [`artifacts/f_train_archive/`](../artifacts/f_train_archive/) |
| **S**-train (0+Top-8) | Closed | [`S_TRAIN_DESIGN.md`](S_TRAIN_DESIGN.md) | [`configs/s_train.json`](../configs/s_train.json) | [`artifacts/s_train_archive/`](../artifacts/s_train_archive/) |

**F** = fewer activated experts · **S** = shared-expert ablation.

Locks and gates for the whole program stay in **this file**. Per-arm hit bars and captions live in the linked DESIGN docs; numbers and analysis live in the REPORT.

## 1. What the paper claims

Fix **total** expert parameters at **16×** a dense FFN, and **activated** expert parameters at **2×** a dense FFN. Sweep MoE *layout* under fixed compute.

| Arm | Layout | Combinations per token |
| --- | --- | ---: |
| GShard | 16 full-size experts, Top-2 | $\binom{16}{2} = 120$ |
| DeepSeekMoE | 64 quarter-size experts: 1 shared + 63 routed; activate 1 shared + Top-7 routed | $\binom{63}{7} \sim 5.5\times 10^8$ |

Paper readout at 2B total / ~0.3B activated, 100B tokens, 8k BPE, internal EN/ZH mix:

- DeepSeekMoE **beats GShard 2B** by a “substantial / overwhelming” margin (same total and activated params).
- DeepSeekMoE **matches GShard ×1.5** (1.5× expert size and expert FLOPs).
- DeepSeekMoE **nearly matches a dense 2B** (same *total* params; the MoE upper bound).
- Pile CE for DeepSeekMoE 2B is **1.808**. Disabling the shared expert and activating one extra routed expert jumps Pile to **2.414**.


---

## 2. Locked 2B skeleton (paper §4.3)

Shared across both MoE arms:

| Item | Paper 2B | This work |
| --- | --- | --- |
| Layers | 9 | 9 |
| $d_\mathrm{model}$ | 1280 | 1280 |
| Heads | 10 × 128 | 10 × 128 |
| Block | not fully specified | V1 block: Pre-Norm RMSNorm, SwiGLU $8/3\,d$, RoPE, MHA (same as [ds-v1-isoflop](../../v1-isoflop/docs/EXPERIMENT_DESIGN.md); intro: [README](../../v1-isoflop/README.md)) |
| MoE placement | every FFN | every FFN |
| Total expert params | 16× dense FFN | 16× dense FFN |
| Activated expert params | 2× dense FFN | 2× dense FFN |
| Init $\sigma$ | 0.006 | 0.006 |
| Optim | AdamW $\beta=(0.9,0.95)$, wd $0.1$ | same |
| LR | $1.08\times 10^{-3}$, 2k linear warmup, ×0.316 at 80% and 90% of steps | same schedule, scaled to the chosen $D$ |
| Grad clip | 1.0 | 1.0 |
| Seq | 2048 | 2048 |
| Global batch | 2048 seq = 4M tokens | 2048 seq = 4M tokens |
| Dropout | none | none |
| Expert-level balance $\alpha_1$ | 0.01 | 0.01 |
| Device-level balance | off (single-device experts) | off |
| Token drop | none | none |
| Tokenizer | 8k BPE | GPT-2 50k (reuse `data_l2`) |
| Data | 100B internal unique | frozen `data_l2`, 40.0B unique |

**GShard (arm A).** 16 experts, each a full SwiGLU FFN. Top-2 softmax routing. Gate value = affinity of the selected experts.

**DeepSeekMoE (arm B).** Each expert is **1/4** of a full FFN (intermediate dim `/4`). Pool = **1 shared + 63 routed**. Every token always runs the shared expert. Router selects Top-7 among the 63. Activated slots = $1+7=8$ quarter-experts = 2× full FFN. Same $\alpha_1=0.01$ on the 63 routed experts only.

That is paper Eq. (9)–(11) with $m=4$, $N=16$, $K=2$, $K_s=1$, so $mK-K_s=7$.

Not in E1: Hash Layer, Switch Transformer, GShard×1.2 / ×1.5, dense 2B upper bound. E3 can add GShard×1.5 if E1 hits.

---

## 3. Why data parallel (EP left unused)

At this skeleton (~2B total / ~0.3B activated), a full replica fits one device in bf16 + Adam, so expert-parallel all-to-all is unnecessary — the paper also kept every expert on one device at 2B. This work locks:

- Data parallel only: full replica per rank (DDP / FSDP no-shard)
- Global batch ~4M tokens via `micro_seqs × 2048 × world × accum` (log `batch_tokens_actual`); closed runs used **`WORLD=20`**
- One job at a time. Do **not** change `world_size` mid-program: that changes the actual batch.

Wall estimates below are order-of-magnitude only (archive used multi-GPU DP).

Estimated wall (order of mag.):

| Phase | Tokens | Steps @ ~4M | Wall / arm |
| --- | ---: | ---: | ---: |
| E0 smoke | 32M | 8 | minutes |
| E1 ranking | 10B | ~2,500 | ~1 day (actual GShard 23.3 h / DSMoE 25.0 h) |
| E2 (optional) | 40B | ~10,000 | multi-day; **incomplete** in archive |

Paper 100B is a different scientific object from E1. `data_l2` has 40.0B unique; 100B would loop 2.5 epochs.

---

## 4. Goals

| # | Goal | Hit | Miss |
| --- | --- | --- | --- |
| C1 | IsoFLOP layout lock | Logged total expert params match to 1%; activated params match to 1%; `batch_tokens_actual` within 10% of 4M | Arms differ in FLOPs or batch |
| C2 | DeepSeekMoE beats GShard | Val **BPB** (byte-normalized, same 50M holdout as IsoFLOP) **lower** for B than A after E1, gap ≥ 0.02 or clearly outside eval scatter | A ≤ B, or either arm routing-collapses (one expert >50% tokens) |
| C3 | Load is not collapsed | Routed expert frequency max/mean < 8 after E1; balance loss finite | Collapse; NaNs |
| C4 | GShard×1.5 (E3) | B **close to** a 1.5×-wider GShard (pre-reg: \|Δ\| ≤ 0.02). Ran even though C2 magnitude missed — claim is independent | Large gap to ×1.5 |

Primary metric is **val BPB**. Downstream few-shot (HellaSwag, PIQA, ARC-e/c) is a **report card** only — it does not gate the program. At 10B on an open mix those scores sit far below paper 100B numbers.

---

## 5. Data

Reuse the frozen IsoFLOP mix (`data_l2`): FineWeb-Edu, Wikipedia EN/ZH, OpenWebMath, Common Crawl WET, Gutenberg. Unique train **39.995B**. Val 50M tokens.

| Phase | $D$ | Epochs vs unique | Loop? |
| --- | ---: | ---: | --- |
| E1 | 10B | 0.25 | no |
| E2 | 40B | 1.00 | no |
| Paper | 100B | n/a | their corpus |

Do not fetch a new mix for E1. A new 8k tokenizer would also break the IsoFLOP val comparison and is not worth it for a ranking check.

---

## 6. Phases

Short definitions only — design links, configs, and archives are in **Status and document map** above.

**E0 — smoke.** Few steps, both arms: forward+backward, expert histograms, no NaN, `unique_train ≥ D`.

**E1 — ranking.** Train GShard and DeepSeekMoE to 10B under matched locks; gate on C1–C3. Canonical archive is the LB-fixed run.

**E2 — longer look (optional).** Continue to 40B only if E1 ranking is noisy but direction is stable. Incomplete in archive.

**E3 — GShard×1.5.** Wider GShard Top-2 vs frozen E1 DeepSeekMoE. See [`E3_DESIGN.md`](E3_DESIGN.md).

**§4.5 — F / S.** F = fewer activated; S = shared ablation. Eval protocols + from-scratch trains: [`ABLATION_ARMS_DESIGN.md`](ABLATION_ARMS_DESIGN.md), [`F_TRAIN_DESIGN.md`](F_TRAIN_DESIGN.md), [`S_TRAIN_DESIGN.md`](S_TRAIN_DESIGN.md). S-eval does **not** substitute for S-train.

Not in this work: aux-loss-free (V3), Hash MoE (V4), expert parallel, MTP, MLA.

---

## 7. What a hit does *not* imply

| Statement | Supported? |
| --- | --- |
| Fine-grained + shared beats matched-FLOP GShard on this mix at 10B | **Yes ranking** (Δ=−0.012); C2 magnitude (≥0.02) **missed** |
| Matches GShard×1.5 | **Yes** (\|Δ\|=0.001) — E3 |
| F-train half-activated clearly beats GShard | **No**; **near-match** (+0.004) |
| Absolute Pile 1.808 / paper Table 1 accuracies | **No** |
| V2 236B or V3 671B would win the same way | **No** |
| Need aux-loss-free to see the gap | **No** (paper 2B used $\alpha_1=0.01$ softmax) |
| Smaller model + more tokens always wins | **No** — this is an architecture IsoFLOP. Out of scope: Chinchilla data-scaling sweeps. |

---

## 8. Implementation notes

Fork the IsoFLOP trainer ([ds-v1-isoflop](../../v1-isoflop/README.md)) rather than Megatron:

- Same training stack, same `data_l2`, same data-parallel recipe (block lock: [v1 DESIGN](../../v1-isoflop/docs/EXPERIMENT_DESIGN.md)).
- New `src/moe.py`: GShard Top-2 and DeepSeekMoE shared+Top-7 on top of the V1 `Block`.
- Log per-step: loss, tok/s, `max_expert_frac`, `balance_loss`.
- Checkpoint `last.pt` periodically; stop with SIGTERM (no `kill -9`).

Runs are sequential under the locked data-parallel recipe — one MoE job at a time.

---

## 9. Pre-registered caption

> On this open mix, at the paper’s 2B skeleton (9L / d=1280, 16× total experts, 2× activated), **DeepSeekMoE (1 shared + Top-7 of 63 quarter-experts) is compared with GShard (Top-2 of 16 full experts)** after 10B tokens. A hit is a **lower val BPB** for DeepSeekMoE with no routing collapse. Absolute paper numbers are out of scope as a target.
