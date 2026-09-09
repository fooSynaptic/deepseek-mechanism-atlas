# Ablation arms — F/S design + status

**Status:** F-eval + S-eval **closed**; F-train **closed** (near-match); **S-train closed** (S-A soft hit) — [`S_TRAIN_DESIGN.md`](S_TRAIN_DESIGN.md).

Paper §4.5 claims that E1 does **not** test: (1) DeepSeekMoE can match GShard with **fewer activated experts**; (2) the **shared expert is not replaceable** by one extra routed expert.

S-eval tests claim (2) via **checkpoint surgery**. **S-train** tests learning from scratch without a shared expert under matched FLOPs — S-eval does not substitute for it.

Full readout: [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md). Reproduce: [`REPRODUCE.md`](REPRODUCE.md).

---

## 1. What the paper actually did

### Arm F — fewer activated experts

Two different protocols in the paper. Do not merge them.

| Protocol | Paper | Train? | Activated vs GShard |
| --- | --- | --- | --- |
| **F-eval (Fig 5)** | Take the trained 1 shared + Top-7 model. At **eval** only, vary routed k from 3 to 7. | No | k=4 already matches GShard Pile |
| **F-train (Fig 6)** | Train from scratch: 1 shared + **Top-3** of 63. | Yes | **Half** the activated expert params; still beats GShard |

Main E1 is **same activated FLOPs** (1+7 vs Top-2). F is a capacity-efficiency claim. It is separate from the Table 1 matched-FLOP ranking.

### Arm S — shared vs extra routed

Paper sentence: disable the shared expert on the **already trained** DeepSeekMoE 2B, activate **one more** routed expert, keep compute. Pile CE **1.808 → 2.414**.

That is **eval-time surgery** on a checkpoint. A from-scratch “no shared” train is a different protocol. A from-scratch 0-shared + Top-8 arm would answer a different question (can the router learn the shared role if it never had one).

| Protocol | Paper | Train? |
| --- | --- | --- |
| **S-eval** | Drop shared, set k_routed = 8, same weights | No |
| **S-train** (optional; absent from paper tables) | 0 shared + 64 routed, Top-8, 10B from scratch | Yes |

---

## 2. Work mapping (executed)

| Job | Protocol | Status | Hit bar | Outcome |
| --- | --- | --- | --- | --- |
| `f_eval_k` | Eval k ∈ {3,4,5,6,7} on `e1_lb_dsmoe` | **Done** | k=4 ≤ same-batch GShard | **Hit** (k≥4) |
| `s_eval_noshare` | Same ckpt, n_shared=0, k=8 | **Done** | large BPB jump vs baseline | **Hit** (+0.135) |
| `f_train_k3` | Train 1+Top-3, 10B | **Done** | BPB &lt; GShard 0.7970 | **Near-match** (+0.004); see [`F_TRAIN_DESIGN.md`](F_TRAIN_DESIGN.md) §8 |
| `s_train_noshare` | 0 shared + Top-8 from scratch | **Done** — 0.7933; S-A +0.008 vs DSMoE | clearly worse than DSMoE 0.7850 (S-A) | see [`S_TRAIN_DESIGN.md`](S_TRAIN_DESIGN.md) |

GShard 10B (0.7970) and DeepSeekMoE 10B (0.7850) are **fixed references**. Do not retrain either for these arms.

S-eval keeps parameter tensors; it only changes which modules run. Activated FLOPs stay ~2× FFN (8 quarter-experts).

---

## 2b. Sequential plan — F/S eval (executed)

**Scope (executed):** F-eval → S-eval; F-train; **S-train** (from-scratch shared ablation). Same data-parallel batch recipe as E1 LB (`WORLD=20`). One MoE job at a time. Data / tokenizer: frozen `data_l2`, GPT-2 50k, same 50M val as E1.

### Why this order

1. **F-eval first** — one ckpt load, five k values; answers “how few routed slots still match GShard.”
2. **S-eval second** — same ckpt, one more forward config; answers “shared off + one extra routed.”
3. F/S eval were **eval-only** on `e1_lb_dsmoe/last.pt`. F-train and **S-train** are separate from-scratch 10B jobs.

### Pipeline status

| Step | Status |
| --- | --- |
| P0 baseline gate | **Done** (0.8204 vs E1 0.7850, within tol 0.05; different val draw) |
| P1 F-eval k=3…7 | **Done** — k=4 already ≤ same-batch GShard |
| P2 S-eval | **Done** — +0.135 BPB when shared off |
| Report | [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) §4.5 |
| Artifacts | `artifacts/fs_eval_archive/` |

### Eval job (executed)

F/S surgery ran as job `fs_eval`; metrics are under `artifacts/fs_eval_archive/`. Training orchestration is not published.

### Decision after P2

| Outcome | Result |
| --- | --- |
| F: k≤4 ≤ GShard; S: large BPB jump | **Hit (direction).** F-train later closed as near-match (+0.004). |

### Explicitly deferred

- Retrain GShard
- Pile train / new tokenizer

F-train closed (near-match): [`F_TRAIN_DESIGN.md`](F_TRAIN_DESIGN.md) §8.  
S-train closed (S-A soft hit): [`S_TRAIN_DESIGN.md`](S_TRAIN_DESIGN.md).

---

## 3. Data: keep `data_l2` or switch to Pile?

Paper 2B numbers are **Pile CE** after **100B** tokens, **8k BPE**, internal EN/ZH mix. This work’s E1 is **val BPB** after **10B** on frozen IsoFLOP `data_l2`, **GPT-2 50k**.

### Stay on `data_l2` (default)

| For | Against |
| --- | --- |
| Same val as E1; F-eval / S-eval can sit on the existing 10B ckpts | Absolute Pile 1.808 / 2.414 literals remain out of scope as raw targets |
| No new tokenizer, mix, or download | Paper Fig 5/6 were not measured in BPB on this mix |
| Ranking vs GShard 0.7970 is apples-to-apples | Downstream (HellaSwag etc.) remains report-card only |
| 10B is already the E1 object; another 10B arm is comparable | Gap sizes will stay smaller than paper 100B Pile |

Use this if the question is: **on the same open mix where E1 already ranked DeepSeekMoE ahead, do §4.5’s two mechanisms still show up in direction?**

### Switch to Pile (only if the question is the paper’s number)

| For | Against |
| --- | --- |
| Metric matches Table 1 / §4.5 (Pile loss) | **Tokenizer mismatch**: 50k GPT-2 vs paper 8k BPE. Same Pile dump still leaves absolute CE out of scope as a target |
| Directly comparable to 1.808 → 2.414 | Paper pretrain mix is **not** Pile; Pile is the **eval**. Training on Pile is a different object than “eval on Pile after internal mix” |
| | New data pipeline, licenses, packing; breaks IsoFLOP val continuity |
| | 100B tokens to chase the published CE is ~10× E1 wall and is **out of E1 scope** |
| | Cannot put Pile CE next to E1 0.7850 BPB without a conversion story |

Training on Pile while evaluating on Pile still misses the paper’s **train mix**. Evaluating E1 checkpoints on a Pile holdout (GPT-2 BPB or CE) is a **third** option: cheap; tokenizer still differs from the paper; useful only as a secondary card.

### Recommendation

| Question | Data |
| --- | --- |
| Do F and S hold **in the same work as E1**? | **`data_l2`**, GPT-2 50k, val BPB. F-eval and S-eval first. |
| Do we need the **1.808 / 2.414** literals? | Do **not** start F/S. That is a new work: 8k tokenizer, paper-like mix or an honest “Pile-only” caption, and likely ≫10B tokens. |
| Extra readout, no new train | Optional **Pile BPB on the existing ckpts** as a report-card row only. |

**Pre-registered caption if `data_l2` is kept**

> On the same 10B `data_l2` runs as E1, eval-only k-sweep and shared-off surgery test paper §4.5 **in direction**. Absolute Pile CE is out of scope as a target. A from-scratch Top-3 arm is a separate 10B train, still on `data_l2`.

---

## 4. What a hit would mean

| Statement | F-eval | F-train | S-eval | S-train |
| --- | --- | --- | --- | --- |
| Fine-grained combo can match GShard with fewer activated slots | yes, if k≤4 | — | — | — |
| Half activated **trained** DeepSeekMoE still beats GShard | — | near-match @ 10B | — | — |
| Shared is not interchangeable with one extra routed expert (**surgery**) | — | — | yes, if BPB jumps | — |
| Shared isolation helps **from scratch** (matched FLOPs) | — | — | — | yes, if clearly worse than 1+Top-7 |

None of these revive C2 (≥ 0.02 BPB) or paper Table 1 accuracies.

---

## 5. Remaining out of scope

Done this program: F/S-eval, F-train, E3, **S-train**. Still skip:

- α sweep, aux-loss-free, expert parallel
- Retraining GShard / full DSMoE for ablation arms
- Optional GShard Top-1 same-activation control for F
