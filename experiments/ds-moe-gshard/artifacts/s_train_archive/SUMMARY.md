# S-train archive

Job `s_train_noshare` (`dsmoe_s0`), 10B tokens, wall **25.7 h** (data-parallel archive).

| Arm | layout | activated / dense | val BPB |
| --- | --- | ---: | ---: |
| DeepSeekMoE 1+Top-7 | 1 shared + Top-7 / 63 | 2.0× | **0.7850** |
| **S-train** | **0 shared + Top-8 / 64** | 2.0× | **0.7933** |
| GShard Top-2 | Top-2 / 16 | 2.0× | 0.7970 |

Δ vs DSMoE = **+0.0083** (S-A hit, barely ≥ 0.008). Δ vs GShard = **−0.0037**.

## Gates

| Gate | Result |
| --- | --- |
| S-A (worse than DSMoE by ≥0.008) | **hit** (+0.0083) |
| S-B (no collapse) | **hit** (`max_frac≈0.26`) |
| S-C soft vs GShard | slightly **better** than GShard |

## Attribution

Shared isolation helps **from scratch**, but the gap is soft. S-eval surgery (+0.135) is **not** interchangeable with S-train: after training with a shared expert, removing it hurts a lot; never having one only costs ~0.008 BPB at 10B.

Full write-up: `docs/EXPERIMENT_REPORT.md`. Design: `docs/S_TRAIN_DESIGN.md`. Reproduce: `docs/REPRODUCE.md`.
