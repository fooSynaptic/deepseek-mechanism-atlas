# Reproduce

This file is the public runbook. The measured tables live in
[`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md) and
`artifacts/published_grid.json`. You do not need to retrain 40B tokens to
rebuild the Formula 4 slope.

## 0. What is being reproduced

DeepSeek-LLM V1 (arXiv:2401.02954) §3 IsoFLOP: fix compute **C = M · D**,
sweep width, read held-out bits-per-byte. This lab is **not** the product 7B/67B
@ 2T run. Absolute BPB is not comparable to the paper (GPT-2 50k vs BBPE ~100k,
open mix vs internal EN/ZH). Ranking within this grid is.

Three layers of reproduce, cheapest first:

1. **Analysis** of the published 41-job table → `a ≈ 0.539`.
2. **Pipeline smoke** (random tokens or a tiny public sample).
3. **Training** one or more grid jobs on Hopper GPUs (96GB HBM).

## 1. Analysis only

From the atlas repo root:

```bash
cd experiments/v1-isoflop
python3 scripts/collect.py
python3 scripts/analyze.py
```

Or run the same two commands with cwd already set to this lab directory.
`analyze.py` skips abandoned / truncated rows (`iso_c3e19_u3` stopped ~29%;
`iso_c3e19_s6` truncated). Valley proxies match report §5:

| C | Proxy | Role |
|--:|-------|------|
| 3e18 | `iso_c3e18_u3` | closed valley on the frozen 40B mix |
| 1e19 | `iso_c1e19_t1` | nearest width to paper M_opt |
| 3e19 | `iso_c3e19_t2` | nearest width to paper M_opt |

Expect log-log OLS **a ≈ 0.539** (paper 0.5243). The fit is illustrative: three
budgets, two proxies on older corpus snapshots.

## 2. Environment

```bash
bash scripts/setup_venv.sh
# or: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Paths default to the repo root. Overrides:

```text
DS_ROOT      repo root
DS_DATA      packed bins (default $DS_ROOT/data)
DS_RESULTS   result.json trees
DS_LOGS      text logs
DS_NPROC     GPUs for run_job / run_wave (local)
HF_ENDPOINT  Hugging Face hub (optional mirror)
```

## 3. Public data / sample preparation

IsoFLOP schedules **D = C / M** tokens per job. The train pool must supply at least
that many **unique** GPT-2 tokens (`unique = sizeof(train.bin) / 2`). Looping D is
refused by `run_wave.py` / `make_grid.py`.

![Reproduce pipeline](figures/reproduce_pipeline.svg)

Sources (official dumps / hub APIs only, no arbitrary-site scrape):

- FineWeb-Edu `sample/10BT` parquet
- Wikipedia EN / ZH (`wikimedia/wikipedia`, 20231101)
- OpenWebMath
- Project Gutenberg
- Common Crawl WET (`data.commoncrawl.org`)
- Optional: Chinese FineWeb-Edu (`--zh-edu-docs`)
- **Skipped in the report mix:** The Stack, C4 (`code-docs=0`, `c4-docs=0`)

```bash
bash scripts/prepare_data.sh smoke     # random uint16, no download
bash scripts/prepare_data.sh demo      # tiny public sample (~2M train)
bash scripts/prepare_data.sh wave-a    # ~4B unique train target
bash scripts/prepare_data.sh full      # report-scale fetch (~40B target)
```

What that wrapper runs:

1. `fetch_open_mix.py --preset …` → `$DS_DATA/raw/mix/*.jsonl` (drop docs shorter than ~80 chars)
2. `tokenize_corpus.py` → `$DS_DATA/train.bin`, `val.bin`, and matching `*.bytes.bin`
   (per-token UTF-8 lengths for bits-per-byte)

| Preset | Approx train / val tokens | Notes |
|--------|---------------------------|-------|
| smoke  | local RNG bins            | no network |
| demo   | 2M / 0.2M                 | pipeline check |
| wave-a | 4B / 50M                  | Wave A / L1 floor |
| full   | 40B / 50M                 | L3 / L4 floor |

The frozen L4 mix used in the report is **39_995_303_545** unique train tokens.
Refetching that mix is optional and slow. Analysis does not need it.

## 4. Grids

```bash
python3 scripts/make_grid.py --wave a    # s1–s6 × {3e18, 1e19, 3e19}
python3 scripts/make_grid.py --wave l1   # t0–t4 × 3e18
python3 scripts/make_grid.py --wave l2   # t0–t4 × 1e19; t2–t4 × 3e19
python3 scripts/make_grid.py --wave l3   # t1, t0 × 3e19
python3 scripts/make_grid.py --wave l4   # u0–u3, drop looping D
```

Each row already has Formula 1 `(batch_tokens, max_lr)` and Formula 2 `M`.
Seed is 1. Sequence length 4096, GPT-2 vocab 50257.

## 5. Local training

One job:

```bash
python3 scripts/run_job.py \
  --job-id iso_c3e18_u0 \
  --grid configs/grid_wave_l4.json \
  --nproc 8 \
  --foreground
```

Sequential queue (skip finished `result.json`, skip looping D):

```bash
python3 scripts/run_wave.py --grid configs/grid_wave_l4.json --nproc 8
python3 scripts/status.py --grid configs/grid_wave_l4.json
```

Smoke without a real corpus:

```bash
bash scripts/smoke.sh
# or
python3 scripts/run_wave.py --grid configs/grid.json --nproc 1 --smoke-steps 8
```

Stop with SIGTERM only: `bash scripts/stop_job.sh <job_id>`.

## 6. Multi-node (optional)

The SSH packer has **no host defaults**. Set aliases and GPU counts yourself:

```bash
export DS_NODES='host-a:8 host-b:8 host-c:8'
python3 scripts/launch_v2.py --grid configs/grid_wave_l.json --max-jobs 3
DS_GRID=configs/grid_wave_l.json bash scripts/install_watch_daemon.sh host-a
```

L3 / L4 rendezvous (nproc may differ across nodes):

```bash
export DS_RDZV_NODES='host-a,4,4,5,6,7,10.0.0.1;host-b,8,0,1,2,3,4,5,6,7,10.0.0.2'
export L3_MASTER_ADDR=10.0.0.1   # or L4_MASTER_ADDR
python3 scripts/launch_l3_20gpu.py --job-id iso_c3e19_t1 --wait
```

Each `DS_RDZV_NODES` entry is `host,nproc,<CUDA_VISIBLE_DEVICES>,ip`. The GPU list may contain commas. See `scripts/lib/rdzv_nodes.py`.

## 7. Collect and fit

```bash
python3 scripts/collect.py --results-dir results --published artifacts/published_grid.json
python3 scripts/analyze.py --results-dir results --published artifacts/published_grid.json
```

Live `results/iso_*/result.json` override published rows with the same `job_id`.
Abandoned / truncated jobs are dropped from the family plot and from the OLS.

## 8. What not to claim

See report §7. In short: this is not a 7B pretrain, not a tokenizer match, and
not an 8-budget Formula 4 refit. The right-arm cliff and a ≈ 0.539 on three
valley proxies are the supported claims.
