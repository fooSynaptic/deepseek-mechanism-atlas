# Reproduce the 10B work

Public runbook. Measured tables: [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md).

Training orchestration is **not** published in this tree. Focus on **what you can check locally**, not on matching the original multi-GPU wall clock.

## What you can reproduce (by machine)

| Machine | In scope here | Out of scope here |
| --- | --- | --- |
| **CPU / laptop** (no GPU) | `scripts/check_arch.py` (param / activation locks); replot curves from `artifacts/*_archive/`; regenerate SVGs under `docs/figures/` | Full 10B train; matching published wall time |
| **Single GPU** | Same as CPU, plus short smoke / forward-backward if you bring your own launcher and a torch env; useful to confirm the replica fits and routing runs | The closed **10B ranking** under the locked ~4M global batch (`WORLD=20` recipe). Changing `world_size` changes the batch — that is a different object |
| **Multi-GPU data parallel** | How the archived 10B runs were executed: full MoE replica per rank, **no** expert parallel, **no** token dropping | Launch scripts / host mesh (unpublished) |

Configs in `configs/*.json` document the recipe for reading; they are not a turnkey cluster submitter. Private `scripts/launch_*.py` / `watch_*.py` (if present) are ops-only and are **not** part of the public reproduce surface.

## Environment

```bash
# from this experiment directory
export DS_ROOT="$PWD"
# optional: DS_PY pointing at a torch env for check_arch / plots
```

## Architecture locks (CPU-friendly)

```bash
python3 scripts/check_arch.py
```

Confirms GShard / DeepSeekMoE / F-train / S-train parameter locks from `src/model.py`.

## Plots from archived metrics (CPU-friendly)

```bash
python3 scripts/plot_lab_family.py
python3 scripts/plot_e1_lb.py
python3 docs/figures/render_lab_overview.py
```

## How the archive was trained (background)

Closed jobs used multi-GPU **data parallel** only (full replica per rank; no expert parallel; no token dropping), with `WORLD=20` for the ~4M-token batch. Treat GPU count / SKU as provenance for `artifacts/`, not as a requirement to re-read the report.

Artifacts: `artifacts/{e1_lb,e3,f_train,s_train,fs_eval}_archive/`.  
Report: `docs/EXPERIMENT_REPORT.md`.
