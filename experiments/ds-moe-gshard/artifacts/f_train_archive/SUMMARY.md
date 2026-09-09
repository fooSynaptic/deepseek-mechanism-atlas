# F-train archive

Job `f_train_k3` (`dsmoe_k3`), 10B tokens, wall **25.1 h** (data-parallel archive).

| Arm | activated / dense | val BPB |
| --- | ---: | ---: |
| GShard Top-2 | 2.0× | 0.7970 |
| F-train 1+Top-3 | **1.0×** | **0.8014** |
| DeepSeekMoE 1+Top-7 | 2.0× | 0.7850 |

Δ vs GShard = **+0.004** (within ~0.015 last-1B scatter).

## Attribution (preferred)

**Near-match GShard** — gap is far softer than full DeepSeekMoE’s edge over GShard (−0.012). Near-match only relative to Fig 6; also far from a large architectural failure.

Primary drivers: **tokens (10B vs paper ~100B)** + **half-activated harder to learn**. Reject a total-param-shortfall story (still 16× dense). No routing collapse (`max_frac≈0.12`).

| Gate | Result |
| --- | --- |
| F-A strict (&lt; GShard) | miss |
| F-B no collapse | hit |
| F-C soft (+≤0.03 vs full DSMoE) | hit (+0.016) |

Full write-up: `docs/EXPERIMENT_REPORT.md`. Family curves: `docs/figures/lab_family_curves.{png,svg}`.
