# E2 archive — incomplete (non-gating)

Attempted 40B continuation after E1. **Not** part of the closed 10B program readout.

| Arm | Status | Notes |
| --- | --- | --- |
| `e2_gshard_w20` | Finished | ~40B tokens, val BPB ≈ **0.680** |
| `e2_dsmoe` | **Incomplete** | meta + `train.jsonl` only (~36B / step ~8980); **no** `result.json` |

Do **not** compare these arms as a paired E2 result. Restart both from aligned ckpts if a 40B ranking is needed later.

Canonical ranking remains E1 LB at 10B (`artifacts/e1_lb_archive/`).
