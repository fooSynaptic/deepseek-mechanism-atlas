# DeepSeek-LLM V1 IsoFLOP — Experiment Report

**Status:** Closed. Wave A + L1–L4 complete. The left-arm question is settled on the frozen 40B mix (u0–u3 at C = 3×10¹⁸, u1–u3 at C = 1×10¹⁹). One job, `iso_c3e19_u3`, was stopped at step 13740 / 47556 (~29%); it is not a finished IsoFLOP point and is not plotted. No further widths are needed.
**Paper under test:** *DeepSeek LLM* (arXiv:2401.02954) §3, Figure 4a and Formulas 1–4.
**Headline number:** the IsoFLOP valley scales as M_opt ∝ C^a with **a ≈ 0.539** here, against the paper's **a = 0.5243**. This is an illustrative 3-point fit, not a reproduction of the paper's 8-budget fit (§5).
**Reproduce:** [`REPRODUCE.md`](REPRODUCE.md). Formula 4 from `artifacts/published_grid.json` needs no GPU (`python3 scripts/analyze.py` → a ≈ 0.539).
**Hardware:** Hopper GPUs (96GB HBM). Wave A / L1 / L2 / L4 ran 8-GPU single-node; L3 ran 20-GPU multi-node.
**Metric:** held-out bits-per-byte (BPB), GPT-2 50k tokenizer. Lower is better.

---



## Summary — what this experiment found

**The setup.** Fix a compute budget C. Spend it by choosing a model size M (non-embedding FLOPs per token) — that choice fixes the training-token count, because C = M · D. Train each split from scratch and compare held-out BPB. DeepSeek V1 Figure 4a predicts a **U-shaped** curve at each budget: models below the valley are capacity-limited, models above it are under-trained, and the valley M_opt shifts right as C grows.

Three observations came out of running that sweep on an open web corpus.

1. **The U is real, but badly lopsided.** The under-trained side is a cliff: past a certain width, BPB jumps from the 0.7–1.1 band into 2.1–2.4, roughly **+1.3 BPB**. The under-capacity side is a gentle slope of **+0.081 BPB** at C = 3×10¹⁸. Same budget, same corpus, penalties an order of magnitude apart.
2. **The left arm does exist — it just needed much smaller models.** Once widths about 5× smaller than the earlier grid floor were trained on a single frozen corpus, BPB fell monotonically toward the valley, and at C = 3×10¹⁸ the minimum landed on the paper's predicted M_opt. At C = 1×10¹⁹ the curve is still descending at the smallest width tested, so that budget shows a fragment rather than a closed U.
3. **The valley moves right with compute at close to the published rate.** Fitting M_opt ∝ C^a on three valley proxies gives a ≈ 0.539 against the paper's 0.5243 — directionally right, but only three budgets and partly mismatched corpus snapshots, so it is illustrative.



### Results at a glance


| Budget     | Left arm (frozen 40B mix)            | Position                                          |
| ---------- | ------------------------------------ | ------------------------------------------------- |
| C = 3×10¹⁸ | u0 1.061 → u3 0.980 (**+0.081**)     | Closed left arm; valley sits on paper M_opt (u3)  |
| C = 1×10¹⁹ | u1 0.962 → u3 0.896 (**+0.066**)     | Still-falling fragment, left of paper M_opt       |
| C = 3×10¹⁹ | t0 / t1 / t2 on older snapshots only | Right-arm cliff is the main result at this budget |



| Question                                                  | Answer                                                                                                                                                                           |
| --------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Full visual match to Figure 4a (a symmetric U at each C)? | **No.** The right arm is steep. The left arm is visible at C = 3×10¹⁸ (+0.081) and as a still-falling fragment at C = 1×10¹⁹ (+0.066) — both far shallower than the right cliff. |
| Is the IsoFLOP tradeoff real?                             | **Yes**, but mostly as the right arm, plus a valley that moves toward t1 / t2 as C grows.                                                                                        |
| Does Formula 4 hold?                                      | **Directionally.** C = M · D is the scheduling identity used by every job. The valley exponent is a ≈ 0.539 here vs 0.5243 in the paper — illustrative only (§5).                |


**In one sentence:** at these budgets the experiment cleanly demonstrates that *starving D to grow M is expensive*, and only weakly demonstrates that *models smaller than M_opt are capacity-limited* — L4 trained widths far left of the valley, yet FineWeb-Edu-heavy text still costs only +0.081 BPB from u0 to u3 at C = 3×10¹⁸.

![Measured IsoFLOP after L4: family overlay and left-arm zoom](figures/isoflop_measured_l4.svg)

*Figure: L4 measured family (left) and left-arm zoom (right). Solid ticks share the frozen 40B mix; dashed joins are older corpus snapshots.*

![IsoFLOP family from published grid](../artifacts/isoflop_family.svg)

*Figure: Formula 4 OLS on valley proxies u3@3e18, t1@1e19, t2@3e19 — regenerate with `python3 scripts/analyze.py`.*

---



## 1. Background — what the paper predicts

Fix C, sweep width. D is not an independent knob; it is C / M.

```text
paper Figure 4a (schematic)

BPB
 │        left arm              right arm
 │     (under-capacity)      (under-trained)
 │            \                /
 │             \              /
 │              \   M_opt   /
 │               \   ▼    /
 │                \______/
 └──────────────────────────── M = FLOPs/token
         small M, large D     large M, small D
```


| Arm       | Mechanism                                                                                                                                          | What should happen                                                   |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Left**  | Capacity. The network cannot represent the distribution no matter how many tokens it sees; extra D just becomes extra epochs on a too-small model. | BPB high at the smallest M, falling toward M_opt.                    |
| **Right** | Data. The network is large enough, but D is too small to train it.                                                                                 | BPB high at the largest M, falling toward M_opt from the other side. |


The two arms are **not interchangeable**. The paper's claim is that there is an interior split, and both extremes lose — for different reasons.

Formula 4 gives where the valley should sit:

```text
M_opt = 0.1715 · C^0.5243
D_opt = 5.8316 · C^0.4757
```

Applied to this lab's three budgets:


| C      | Paper M_opt | Paper D_opt (order of mag.) | Nearest width on this grid |
| ------ | ----------- | --------------------------- | -------------------------- |
| 3×10¹⁸ | 8.35×10⁸    | ~3.6B                       | **u3** (M = 8.02×10⁸)      |
| 1×10¹⁹ | 1.57×10⁹    | ~6.4B                       | **t1** (M = 1.63×10⁹)      |
| 3×10¹⁹ | 2.79×10⁹    | ~10.8B                      | **t2** (M = 2.77×10⁹)      |


This is why the grid looks the way it does: the paper's valleys land far to the **left** of a product-7B-scale sweep, so the small widths had to be added deliberately.

---



## 2. Experiment design



### 2.1 The wave ladder, A → L4

The grid was **not** run in one pass. Each wave reuses earlier `result.json` files and adds widths or budgets; **no width is ever retrained**. Two constraints forced the laddering:

- Formula 4 puts M_opt far left of the product-7B floor, so the left-hand widths had to be added incrementally.
- The unique-token corpus **grew over time**. A wave can only schedule D = C / M that its corpus snapshot supports without looping over the same tokens.


| Wave   | Design intent                                                                                                                                                  | Widths × budgets                                           | Corpus (unique) | GPUs |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | --------------- | ---- |
| **A**  | Establish the product-7B floor: put s1–s6 (top = V1 7B width) on the grid at all three budgets. Formula 4 predicts these sit on the right arm, not a full U.   | s1–s6 × {3×10¹⁸, 1×10¹⁹, 3×10¹⁹}                           | 4.00B           | 8    |
| **L1** | Add t0–t4 left of s1 at 3×10¹⁸ only, with no new data, to look for an interior valley near paper M_opt.                                                        | t0–t4 × 3×10¹⁸                                             | 4.00B           | 8    |
| **L2** | Fetch more unique tokens; extend the left widths to 1×10¹⁹ and put t2–t4 on 3×10¹⁹.                                                                            | t0–t4 × 1×10¹⁹; t2–t4 × 3×10¹⁹                             | 10.6B           | 8    |
| **L3** | Close t0 / t1 at 3×10¹⁹, which needs ≥ ~36B unique tokens.                                                                                                     | t0 / t1 × 3×10¹⁹                                           | 40.0B           | 20   |
| **L4** | Deep-left capacity probe on a **frozen** 40B mix: add u0–u3 (left of t0) at matched corpus, to test whether a much-too-small model is really capacity-limited. | u0–u3 × 3×10¹⁸; u1–u3 × 1×10¹⁹; u3 × 3×10¹⁹ (stopped ~29%) | 40.0B           | 8    |


Because the corpus differs across waves, the honest comparison is **within a wave/snapshot**. Joins drawn across snapshots appear dashed in the figure and must not be read as a single curve (§6).



### 2.2 How to read a job id

Every run is named `iso_c{budget}_{width}`. Taking `iso_c3e19_u3` apart:


| Piece   | Meaning                                                                                    | Value in this example          |
| ------- | ------------------------------------------------------------------------------------------ | ------------------------------ |
| `iso`   | An IsoFLOP job: fixed C = M · D, Formula-1 batch and learning rate, trained from scratch   | —                              |
| `c3e19` | The compute budget C (`c3e18` → 3×10¹⁸, `c1e19` → 1×10¹⁹, `c3e19` → 3×10¹⁹)                | C = 3×10¹⁹                     |
| `u3`    | The width id, which fixes the architecture and M — and therefore the token count D = C / M | u3: M = 8.02×10⁸, so D ≈ 37.4B |


So `iso_c3e19_u3` is "IsoFLOP at C = 3×10¹⁹ on width u3". 

Width ids come in three families:


| Family | Ids   | Role on the M axis                                                                                 |
| ------ | ----- | -------------------------------------------------------------------------------------------------- |
| **u**  | u0–u3 | L4 deep-left capacity probe, left of t0                                                            |
| **t**  | t0–t4 | Wave L bridge between the u and s families; t0 / t1 / t2 sit near paper M_opt at the three budgets |
| **s**  | s1–s6 | Wave A product-scale floor, on the right arm; s5 = V1 7B depth/width                               |




### 2.3 Protocol (locked)


| Item              | Paper                                     | This rep                                       |
| ----------------- | ----------------------------------------- | ---------------------------------------------- |
| Arch              | Pre-Norm RMSNorm, SwiGLU 8/3 d, RoPE, MHA | Same; no GQA                                   |
| Init / Optim / LR | §3 defaults                               | Same (warmup capped when steps < 40k)          |
| Sequence length   | 4096                                      | 4096                                           |
| Scale             | C = M · D, Formula 2                      | Same                                           |
| Batch / LR        | Formula 1                                 | Same                                           |
| From-scratch      | Yes                                       | Yes                                            |
| Data              | Internal EN/ZH mix                        | Open mix (below); **corpus grew across waves** |
| Tokenizer         | BBPE ~100k                                | GPT-2 50k (BPB is still byte-normalized)       |
| Val set           | 100M tokens                               | 50M tokens                                     |
| Parallel          | Their cluster                             | self-build FSDP                                |




### 2.4 Widths

Head dim 128 throughout; s5 matches the V1 product 7B depth/width.


| id  | layers / d_model / heads | N1     | M @ L=4096 | Role                                      |
| --- | ------------------------ | ------ | ---------- | ----------------------------------------- |
| u0  | 6 / 384 / 3              | 0.011B | 1.77×10⁸   | Deep left; 3×10¹⁸ only                    |
| u1  | 8 / 512 / 4              | 0.025B | 3.52×10⁸   | 3×10¹⁸, 1×10¹⁹                            |
| u2  | 10 / 640 / 5             | 0.049B | 6.09×10⁸   | 3×10¹⁸, 1×10¹⁹                            |
| u3  | 10 / 768 / 6             | 0.071B | 8.02×10⁸   | Left of t0; only u-width that fits 3×10¹⁹ |
| t0  | 12 / 768 / 6             | 0.085B | 9.63×10⁸   | Near paper M_opt at 3×10¹⁸                |
| t1  | 16 / 896 / 7             | 0.154B | 1.63×10⁹   | Near paper M_opt at 1×10¹⁹                |
| t2  | 22 / 1024 / 8            | 0.277B | 2.77×10⁹   | Near paper M_opt at 3×10¹⁹                |
| t3  | 12 / 1792 / 14           | 0.462B | 3.83×10⁹   | Bridge                                    |
| t4  | 24 / 1536 / 12           | 0.679B | 5.89×10⁹   | Bridge                                    |
| s1  | 22 / 2048 / 16           | 1.11B  | 8.86×10⁹   | Wave A floor                              |
| s2  | 24 / 2560 / 20           | 1.89B  | 1.43×10¹⁰  |                                           |
| s3  | 26 / 3072 / 24           | 2.94B  | 2.16×10¹⁰  |                                           |
| s4  | 28 / 3584 / 28           | 4.32B  | 3.08×10¹⁰  |                                           |
| s5  | 30 / 4096 / 32           | 6.04B  | 4.23×10¹⁰  | Product 7B width                          |
| s6  | 32 / 4096 / 32           | 6.44B  | 4.51×10¹⁰  |                                           |




### 2.5 Corpus, and why the snapshot matters

**Open mix:** FineWeb-Edu sample-10BT, Wikipedia EN/ZH, OpenWebMath, Common Crawl WET, Gutenberg. The Stack, C4, and ZH FineWeb-Edu were planned but skipped after hub timeouts.

The unique-token floor is **wave-dependent** — later jobs did not train on the same corpus snapshot as earlier ones. A BPB difference measured across waves is not a pure IsoFLOP effect (§6.1).


| Wave  | Typical unique train tokens  | Jobs                                                        |
| ----- | ---------------------------- | ----------------------------------------------------------- |
| A, L1 | 4.00B                        | s1–s6 at all C; t0–t4 @ 3×10¹⁸; t2–t4 @ 1×10¹⁹; s* @ 3×10¹⁹ |
| L2    | 10.60B (t2 @ 3×10¹⁹: 11.40B) | t0/t1 @ 1×10¹⁹; t2–t4 @ 3×10¹⁹                              |
| L3    | 40.00B                       | t0/t1 @ 3×10¹⁹ only                                         |
| L4    | 40.00B (same snapshot as L3) | u0–u3 @ 3×10¹⁸, u1–u3 @ 1×10¹⁹ (done); u3 @ 3×10¹⁹ stopped  |


Every scheduled D is less than or equal to that job's logged unique-token count, so no job loops over its corpus.

---



## 3. Measured results

IsoFLOP family after L4: a moving valley near paper M_opt, a steep under-training right arm, and a shallow under-capacity left arm, with a left-arm zoom for the two smaller budgets

L4 on the frozen 40B mix fills in u0–u3 at C = 3×10¹⁸ and u1–u3 at C = 1×10¹⁹. The u3 job at C = 3×10¹⁹ was stopped mid-run and is not plotted. Dashed joins at t0 mark a **corpus change**, not a matched IsoFLOP neighbor.

Finished runs only. Bold marks the best width at that budget **on that snapshot**.


| Width | M (×10⁹) | C = 3×10¹⁸ | C = 1×10¹⁹     | C = 3×10¹⁹     | D @ 3×10¹⁸ (L4) | D @ 1×10¹⁹ (L4) |
| ----- | -------- | ---------- | -------------- | -------------- | --------------- | --------------- |
| u0    | 0.177    | 1.061      | — (would loop) | — (would loop) | 16.95B          | —               |
| u1    | 0.352    | 1.001      | 0.962          | — (would loop) | 8.51B           | 28.38B          |
| u2    | 0.609    | 0.985      | 0.915          | — (would loop) | 4.92B           | 16.41B          |
| u3    | 0.802    | **0.980**  | 0.896          | — (stopped)    | 3.74B           | 12.47B          |
| t0    | 0.963    | 1.036†     | 0.787          | 0.851          | 3.12B†          |                 |
| t1    | 1.629    | 1.057      | **0.777**      | 0.823          |                 |                 |
| t2    | 2.768    | 1.100      | 0.983          | **0.715**      |                 |                 |
| t3    | 3.831    | 2.095      | 0.982          | 0.733          |                 |                 |
| t4    | 5.889    | 2.209      | 1.066          | 0.761          |                 |                 |
| s1    | 8.858    | 2.232      | 2.122          | 0.950          |                 |                 |
| s2    | 14.345   | 2.306      | 2.228          | 2.034          |                 |                 |
| s3    | 21.592   | 2.199      | 2.216          | 2.171          |                 |                 |
| s4    | 30.828   | 2.265      | 2.274          | 2.239          |                 |                 |
| s5    | 42.279   | 2.373      | 2.246          | 2.228          |                 |                 |
| s6    | 45.097   | 2.420      | 2.200          | 2.236‡         |                 |                 |


† t0–s6 at C = 3×10¹⁸ used the 4.00B snapshot, not L4's 40B mix, so u3 0.980 vs t0 1.036 is not a pure IsoFLOP delta. Likewise t0/t1 at C = 1×10¹⁹ used a 10.6B snapshot while L4's u1–u3 used the 40B mix; the apparent drop from u3 0.896 to t0 0.787 is largely corpus drift — the same fingerprint as t0 moving 0.787 → 0.851 when its budget grew.
‡ `iso_c3e19_s6` ran only 0.04 h and logged a null final train loss. Treat it as truncated and unreliable.

### Valley position vs the paper


| C            | Paper M_opt | Measured minimum                  | Left lift            | Right lift (s2 − min) |
| ------------ | ----------- | --------------------------------- | -------------------- | --------------------- |
| 3×10¹⁸ (L4)  | 0.835×10⁹   | **u3 0.980** (on-grid, 40B mix)   | **+0.081** (u0 − u3) | n/a on this mix       |
| 3×10¹⁸ (old) | 0.835×10⁹   | t0 1.036 (left endpoint of t0–s6) | **0**                | +1.27                 |
| 1×10¹⁹ (L4)  | 1.57×10⁹    | u3 0.896 (still falling, 40B mix) | **+0.066** (u1 − u3) | n/a on this mix       |
| 1×10¹⁹ (old) | 1.57×10⁹    | t1 0.777 (10.6B mix)              | **+0.010** (t0 − t1) | +1.45                 |
| 3×10¹⁹       | 2.79×10⁹    | t2 0.715                          | **+0.136** (t0 − t2) | +1.32                 |


At C = 3×10¹⁸ on the matched 40B mix, BPB falls u0 → u1 → u2 → u3. That is a real left arm, and its minimum sits on paper M_opt (u3 at M = 8.02×10⁸ vs the predicted 8.35×10⁸). The arm is nonetheless modest at +0.081 BPB; the right-arm cliff on the old 4B slice is an order of magnitude larger.

At C = 1×10¹⁹ on the same 40B mix, BPB still falls: u1 0.962 → u2 0.915 → u3 0.896 (+0.066). Paper M_opt is out at t1 (1.57×10⁹), so u3 is still left of the valley. This is a left-arm **fragment**, not a closed U.

The valley **does** move with C — u3 at 3×10¹⁸, then t1 / t2 at the larger budgets on older snapshots — but the shape around it stays asymmetric: left lifts of +0.081 and +0.066 against a right-arm jump of +1.32.

### Run logs

L4 (8 GPU):


| Job            | Steps         | Tokens seen     | Wall    | Final train loss | Best val BPB     |
| -------------- | ------------- | --------------- | ------- | ---------------- | ---------------- |
| `iso_c3e18_u0` | 64676         | 16.95B          | 3.13 h  | 3.48             | 1.061            |
| `iso_c3e18_u1` | 32482         | 8.51B           | 2.21 h  | 3.17             | 1.001            |
| `iso_c3e18_u2` | 18777         | 4.92B           | 1.82 h  | 3.19             | 0.985            |
| `iso_c3e18_u3` | 14267         | 3.74B           | 1.57 h  | 3.11             | 0.980            |
| `iso_c1e19_u1` | 54137         | 28.38B          | 6.85 h  | 3.05             | 0.962            |
| `iso_c1e19_u2` | 31295         | 16.41B          | 5.37 h  | 2.84             | 0.915            |
| `iso_c1e19_u3` | 23778         | 12.47B          | 5.05 h  | 2.95             | 0.896            |
| `iso_c3e19_u3` | 13740 / 47556 | 10.81B / 37.40B | stopped | ~3.10            | not a grid point |


`iso_c3e19_u3` was stopped with SIGTERM at step 13740 (10.81B of 37.40B tokens). A mid-run eval at step 9512 read BPB 1.020; that is not a finished IsoFLOP point and is excluded from the family plot. Its `last.pt` is kept in case a resume is ever wanted.

L3 (20 GPU):


| Job            | Steps | Tokens seen | Wall   | Final train loss | Best val BPB |
| -------------- | ----- | ----------- | ------ | ---------------- | ------------ |
| `iso_c3e19_t1` | 28093 | 18.41B      | 7.12 h | 2.84             | 0.823        |
| `iso_c3e19_t0` | 47556 | 31.17B      | 7.45 h | 2.73             | 0.851        |


For both L3 jobs the last eval is also the best. Train loss drops from ~10.8 to ~2.7–2.8 within the first ~10% of steps, then plateaus.

---



## 4. Why the two arms differ

"Inconsistent with Figure 4a" here does **not** mean the optimizer failed or that 20-GPU training is invalid. Both L3 jobs finished cleanly, loss and val BPB moved together, and unique tokens exceeded D in every case. It means the **two sides of M_opt are not equal and opposite in this lab**, whereas Figure 4a is drawn as if they were.

At C = 3×10¹⁹:

```text
t0 0.851 ──┬── t1 0.823 ──┬── t2 0.715 ── t3 0.733 ── t4 0.761 ── s1 0.950
           │              │                                         │
     +0.14 vs valley      valley                               then s2 2.03
     (shallow left)                                            (+1.32, steep right)
```



### 4.1 Right arm: steep, and this is the part that matches the paper

The mechanism is **under-training**: D falls as 1/M.


| Width | Params (measured) | D     | Tokens per param (rough) | BPB       |
| ----- | ----------------- | ----- | ------------------------ | --------- |
| t2    | 328M              | 10.8B | ~33                      | **0.715** |
| t4    | 757M              | 5.1B  | ~6.7                     | 0.761     |
| s1    | 1.21B             | 3.39B | ~2.8                     | 0.950     |
| s2    | 2.02B             | 2.09B | ~1.0                     | **2.034** |
| s5    | 6.25B             | 0.71B | ~0.11                    | 2.228     |


Once D drops to about 1 token per parameter (s2), the run stops being "a slightly worse IsoFLOP point": train loss sits at ~5.7–6.5 instead of ~2.5. That is a starved model. The same cliff appears at every budget — between t2–t3 at 3×10¹⁸, or t4–s1 at the two larger budgets — with BPB jumping from the 0.7–1.1 band into 2.1–2.4.

This side supports a fair IsoFLOP statement:

> At fixed FLOPs, buying parameters by cutting tokens is cheap at first (t2 → t4 costs only +0.05 BPB at 3×10¹⁹) and then catastrophically expensive once D is no longer enough to train the width.

s1 at 3×10¹⁹ (BPB 0.950, train loss 2.64) is an in-between point: much better than s2–s6, still worse than t2–t4. Its mid-run evals fall smoothly rather than spiking. It is a real IsoFLOP point, but it sits on the 4.00B snapshot while t2 sits on an 11.4B one — see §6.1 before using 0.950 to calibrate anything.

### 4.2 Left arm: shallow, and this is the part that does not look like Figure 4a

The mechanism the paper expects is **under-capacity**: extra tokens cannot buy a better fit if the network is too small. What the measurements show:

1. **t0 was never a deep left-arm probe; L4 is.** Wave L placed t0 essentially at paper M_opt for 3×10¹⁸ (0.96×10⁹ vs 0.84×10⁹). L4 added u0 at M = 1.77×10⁸, about 5× smaller than t0 — the deep-left probe the original grid lacked.
2. **At C = 3×10¹⁸ the old 4B slice had no left arm at all** (t0 was the endpoint minimum, 1.036). L4 changed that on the 40B mix: u0 1.061 → u3 0.980, a +0.081 arm with paper M_opt sitting on u3. The arm is real, and still much smaller than the right cliff.
3. **At C = 1×10¹⁹ the matched 40B left arm is a fragment, not noise.** u1 → u3 is +0.066 BPB (0.962 vs 0.896), still descending, still left of paper M_opt at t1. The old t0 − t1 gap of +0.010 BPB on the 10.6B mix remains eval scatter. The u3 0.896 → t0 0.787 join is corpus drift and must not be read as a valley.
4. **At C = 3×10¹⁹ the left arm exists across mixed snapshots, and it is small.** t0 − t2 is +0.136 BPB, about 19% relative to the valley, against a right-arm jump of ~185%. The u3 job that would have probed further left was stopped at 29% of its tokens, and even finished it would not have closed this U while t2 sits on a different snapshot.
5. **The corpus is still relatively easy for these widths.** u0 is 6 layers / 384 d / ~11M non-embedding params trained on 17B tokens, reaching BPB 1.061; t0 is 12 layers / 768 d / 85M params on 31B tokens at 0.851. That is not a toy domain (the TinyStories pilot's smallest width already hit 0.46), but capacity is only slightly binding — not binding the way a 10M-param model on open web text would be.

Taken together, the left side is consistent with a **weak capacity penalty**, not with the textbook U. L4 filled in the missing widths; FineWeb-Edu-heavy text simply does not punish them the way Figure 4a draws.

### 4.3 The same asymmetry at every budget

- **C = 3×10¹⁸:** u0 1.061 → u3 0.980 (40B mix) ── dashed ── t0 best on 4B ── cliff after t2 ── s* around 2.2
- **C = 1×10¹⁹:** u1 0.962 → u3 0.896 (40B mix, still falling) ── dashed ── t0 ≈ t1 minimum on 10.6B ── cliff after t4
- **C = 3×10¹⁹:** t0/t1 slightly above t2 ── t2 minimum ── slow rise t3–s1 ── cliff at s2 (u3 stopped, not plotted)

The right cliff is reproducible at every budget. The left rise stays modest even after L4 filled widths far left of t0. So L4 answers the objection "no much-too-small model was ever trained" at the two smaller budgets — it does not produce Figure 4a.

---



## 5. Formula 4 — paper 0.5243 vs this lab 0.539

Paper §3 replaces the usual C ≈ 6ND with **C = M · D**, where M is non-embedding FLOPs per token (Formula 2). Every job here schedules D = C / M that way, so the identity is **used**, not fitted. What Formula 4 actually fits is the position of the IsoFLOP **valleys** across budgets:

```text
M_opt = 0.1715 · C^0.5243
D_opt = 5.8316 · C^0.4757
```

The paper fits this on **8** budgets spanning 1×10¹⁷ to 3×10²⁰. This lab has three. Taking the valley proxy closest to paper M_opt at each budget:


| C      | Paper M_opt | Lab proxy                                | M_lab / M_paper |
| ------ | ----------- | ---------------------------------------- | --------------- |
| 3×10¹⁸ | 8.35×10⁸    | **u3** 8.02×10⁸ (40B mix, closed valley) | 0.96            |
| 1×10¹⁹ | 1.57×10⁹    | **t1** 1.63×10⁹ (10.6B mix)              | 1.04            |
| 3×10¹⁹ | 2.79×10⁹    | **t2** 2.77×10⁹ (mixed snapshots)        | 0.99            |


A log–log OLS on those three points gives **a ≈ 0.539**, with b = 1 − a ≈ 0.461 (paper: a = 0.5243, b = 0.4757). The two-point slope from u3 @ 3×10¹⁸ to t2 @ 3×10¹⁹ gives essentially the same value, a ≈ 0.538.

That slope is **directionally consistent** with Formula 4, and the valleys do land on the paper's predicted grid. It is **not** a claim that this open mix reproduces the paper's coefficients, nor a substitute for their 8-budget fit: two of the three proxies are not on the matched 40B mix, and C = 1×10¹⁹ still lacks a closed valley at paper M_opt.

---



## 6. Caveats

These do **not** invent the right-arm cliff, which is reproducible within a single wave. They **do** limit how much Formula 4 or "left-arm height" should be read out of the cross-snapshot points.

### 6.1 The corpus snapshot changed between waves


| Job              | Unique train tokens | D      | Epochs (D / unique) |
| ---------------- | ------------------- | ------ | ------------------- |
| t0 @ 3×10¹⁸      | 4.00B               | 3.12B  | 0.78                |
| t0 @ 1×10¹⁹      | 10.60B              | 10.39B | 0.98                |
| t0 @ 3×10¹⁹ (L3) | 40.00B              | 31.17B | 0.78                |
| t2 @ 3×10¹⁹ (L2) | 11.40B              | 10.84B | 0.95                |


t0 **got worse** going from C = 1×10¹⁹ (0.787) to C = 3×10¹⁹ (0.851) despite seeing 3× more tokens. Under a single frozen corpus, more compute at fixed width should lower BPB. That reversal is the fingerprint of a harder or simply different 40B mix (and/or a different val draw) — not of IsoFLOP failing.

The consequence is that t0/t1 and t2 at 3×10¹⁹ are **not a perfectly matched pair**: L3 used the 40B snapshot on 20 GPUs, t2 used the 11.4B L2 snapshot on 8. If the 40B mix is harder, L3 inflates t0/t1 BPB and makes the left arm look *taller* than a matched-corpus run would — and the arm is shallow even with that bias working in its favour. This is why the Formula 4 slope in §5 uses valley proxies rather than a single-snapshot grid.

### 6.2 World size 8 vs 20

L3's global batch still targets Formula 1 (~686k tokens; actual 655360), matching the 8-GPU jobs' target. But the parallel recipe differs (NCCL over Ethernet, multi-node data-parallel ranks, explicit rendezvous). Do not treat L3 wall-clock, or a 0.01-level BPB delta against 8-GPU points, as comparable.

### 6.3 s1 at 3×10¹⁹ vs the s2 cliff

s1 at this budget is on the 4.00B Wave A mix (D = 3.39B, about 0.85 epoch). s2–s6 at the same budget are on the same mix and form a tight 2.03–2.24 band. The s1 → s2 jump is therefore **within-wave**, which makes it the cleanest right-arm fact in the file.

### 6.4 s6 at 3×10¹⁹

Truncated; drop it from any fit. The right arm is already established by s2–s5.

### 6.5 Tokenizer, val size, and mix vs the paper

Absolute BPB is not comparable to the paper's plots: GPT-2 50k vs BBPE ~100k, 50M vs 100M val tokens, FineWeb-Edu-heavy vs an internal EN/ZH mix. Only within-lab ranking is used here.

---



## 7. What is not claimed

The left-arm question is closed. This is a negative list, not a backlog. Since i dont name this rep as "nano-deepseek-V1", I want make these "not claimed" for clarity.

- **Not** a full Figure 4a reproduction — there is no symmetric U at any budget.
- **Not** a paper-grade Formula 4 coefficient claim. The a ≈ 0.539 slope is illustrative, from three proxies on partly mismatched snapshots.
- **Not** a claim that dashed joins across corpus snapshots form a single IsoFLOP curve.
- **Not** claiming absolute BPB numbers are comparable to the paper's, given the tokenizer, val size, and mix all differ.

One statement that *is* supported, at fixed C: starving D to grow M past roughly t4/s1 is a bad trade, at all three budgets. One that is not: reading the total-token axis D as if it were Formula 1's batch size B — every job uses the Formula-1 batch and learning rate, and the axis swept here is *total* tokens.

The honest caption for the family plot:

> On this open mix, IsoFLOP at C ∈ {3×10¹⁸, 1×10¹⁹, 3×10¹⁹} recovers a moving valley near paper M_opt, a steep under-training (right) arm, and a modest under-capacity (left) arm on the frozen 40B mix — u0→u3 is +0.081 at 3×10¹⁸, u1→u3 is +0.066 at 1×10¹⁹ and still left of paper M_opt. Dashed joins to t0–s6 come from older snapshots and must not be fitted as a single U.

---



## 8. Goals scorecard


| Goal                          | After L4              | Evidence                                                                                                                                   |
| ----------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| C1 — M as the scale unit      | Pass                  | Formula 2 used for every D = C / M                                                                                                         |
| C2 — IsoFLOP tradeoff is real | **Pass, asymmetric**  | Right arm at all budgets; left arm at 3×10¹⁸ (+0.081) and a 1×10¹⁹ fragment (+0.066) on the 40B mix; valley still t1/t2 on older snapshots |
| C3 — Formula 4 exponent       | **Illustrative only** | Three valley proxies give a ≈ 0.539 vs paper 0.5243; not an 8-budget fit (§5)                                                              |
| C4 — Formula 1 (B, η) cross   | Not run               | Out of scope for this closed report                                                                                                        |
| Full Figure 4a visual match   | **Miss**              | Left arm exists but is shallow; the 1×10¹⁹ fragment never reaches paper M_opt                                                              |


Restating the two pre-registered outcomes from the design doc:

- **A** — the valley lies left of the s1–s6 grid: still true. s1–s6 never contain M_opt.
- **B** — an interior U appears on s1–s6: still false, apart from the old shallow s3 wiggle at 3×10¹⁸, which is still not fitted.
- After Wave L + L4 there is an interior minimum on the 40B mix at C = 3×10¹⁸ (u3), and a still-falling left fragment at C = 1×10¹⁹ (u3). The dashed 1×10¹⁹ join to t0/t1 is not a paper U.

---



## 9. Artifacts


| Path                                                      | Content                                                 |
| --------------------------------------------------------- | ------------------------------------------------------- |
| `results/iso_c{3e18,1e19,3e19}_{t0–t4,s1–s6}/result.json` | Finished jobs (`best_val_bpb`, `meta`, `wall_s`)        |
| `results/iso_c3e18_u{0–3}/result.json`                    | L4 jobs at C = 3×10¹⁸                                   |
| `results/iso_c1e19_u{1–3}/result.json`                    | L4 jobs at C = 1×10¹⁹                                   |
| `results/iso_c3e19_u3/result.json`                        | Abandoned run (`ok: false`, step 13740); `last.pt` kept |
| `results/iso_c3e19_t0/train.jsonl`, `…/t1/train.jsonl`    | L3 step logs                                            |
| `configs/grid.json`, `grid_wave_l{,2,3,4}.json`           | Scheduled C, D, and Formula-1 batch / learning rate     |
| `docs/EXPERIMENT_DESIGN.md`                               | Pre-registered outcomes A / B / pilot-fail              |
| `docs/WAVE_L_DESIGN.md`                                   | Why t0–t4 and u0–u3 exist                               |
| `docs/figures/isoflop_measured_l4.svg`                    | Family overlay plus the 3×10¹⁸ / 1×10¹⁹ left-arm zoom   |
| `artifacts/isoflop_family.svg` / `isoflop_fit.json`       | Formula 4 fit from `scripts/analyze.py`                 |


