# E1 archive (lightweight)

Metrics / logs only. Large checkpoints are not bundled in this archive.

## Primary ranking (matched data-parallel recipe)

| job_id | arch | world | accum | steps | val BPB | wall_s | mean tok/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `e1_gshard_w20` | gshard | 20 | 98 | 2492 | **0.7964** | 81016 | ~124k |
| `e1_dsmoe` | dsmoe | 20 | 98 | 2492 | **0.7933** | 88039 | ~114k |

- ΔBPB (dsmoe − gshard) ≈ **−0.003** (dsmoe slightly better).
- C1 layout lock: hit (`expert_over_dense=16`).
- C2 (≥0.02 gap): **miss**.
- C3 (no collapse): **miss** — both arms `max_expert_frac≈0.99–1.0` throughout.

## Legacy / non-matched

| job_id | world | val BPB | note |
| --- | ---: | ---: | --- |
| `e1_gshard` | 24 | 0.7854 | different accum/steps; unfair vs w20 dsmoe |

## Layout in this archive

```text
e1_archive/
  SUMMARY.md
  e1_gshard/{result,meta,train}
  e1_gshard_w20/{result,meta,train}
  e1_dsmoe/{result,meta,train}
  smoke/*.smoke24/{result,meta,train,...}
```
