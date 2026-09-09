# Wave L — left-of-s1 IsoFLOP supplement

Wave A (s1–s6) sat on the **right arm** of paper Figure 4a: Formula 4
`M_opt` at `C ∈ {3e18, 1e19, 3e19}` is `~8.35e8 / 1.57e9 / 2.79e9`, while
s1 already has `M ≈ 8.86e9`. This wave adds widths **left of s1**. Do not
retrain s1–s6; stitch Wave A `result.json` into the combined curves.

## Unique-token floors (`D = C / M`)

Current corpus: **4.00B** unique train tokens.

| C | Min M without looping |
|--:|----------------------:|
| 3e18 | ≥ 7.5e8 |
| 1e19 | ≥ 2.5e9 |
| 3e19 | ≥ 7.5e9 (≈ s1) |

## Phases

1. **L1 (now, no new data):** t0–t4 × **C=3e18 only**.
2. **L2 (≥12B unique train):** t0–t4 × C=1e19, and widths with `M ≥ 2.8e9` × C=3e19 (t2–t4; t2 only if unique ≥ 11B).
3. **L3 (~36B):** t0/t1 × C=3e19 (done).
4. **L4 (no new fetch):** widths **left of t0** on the frozen ~40B unique set. Drop any `(C, width)` whose `D = C/M` would loop.

Do **not** start L2 fetch until L1 shows a non-pilot interior valley.

## Widths (head dim 128)

| id | n_layer / d / heads | N1 | M @ 4096 | Role |
|----|---------------------|---:|---------:|------|
| t0 | 12 / 768 / 6 | 0.085B | 9.63e8 | Near `M_opt@3e18` |
| t1 | 16 / 896 / 7 | 0.154B | 1.63e9 | Near `M_opt@1e19` |
| t2 | 22 / 1024 / 8 | 0.277B | 2.77e9 | Near `M_opt@3e19` |
| t3 | 12 / 1792 / 14 | 0.462B | 3.83e9 | Bridge toward s1 |
| t4 | 24 / 1536 / 12 | 0.679B | 5.89e9 | Upper bridge |
| u0 | 6 / 384 / 3 | 0.011B | 1.77e8 | Deep left; C=3e18 only |
| u1 | 8 / 512 / 4 | 0.025B | 3.52e8 | C=3e18, 1e19 |
| u2 | 10 / 640 / 5 | 0.049B | 6.09e8 | C=3e18, 1e19 |
| u3 | 10 / 768 / 6 | 0.071B | 8.02e8 | Left of t0; only L4 width that fits C=3e19 |

Protocol matches Wave A: Formula 1 `(B, η)`, `L=4096`, from-scratch, 8-GPU
single-node FSDP, `--no-checkpoint`.

## L1 jobs (5)

`iso_c3e18_t0` … `iso_c3e18_t4`. Grid file: `configs/grid_wave_l.json`.

## L4 jobs (8, frozen 40B unique)

No new corpus. `D` must be `≤ 39.995B`. Grid: `configs/grid_wave_l4.json`.

| job | D | Why this C | Status |
|-----|--:|------------|--------|
| `iso_c3e18_u0` … `u3` | 17.0B / 8.5B / 4.9B / 3.7B | All four widths fit | **done** — BPB 1.061 / 1.001 / 0.985 / **0.980** |
| `iso_c1e19_u1` … `u3` | 28.4B / 16.4B / 12.5B | u0 @ 1e19 would loop (56B) | **done** — BPB 0.962 / 0.915 / 0.896 |
| `iso_c3e19_u3` | 37.4B | Only u3 is above `M ≥ C/unique ≈ 7.5e8` | **stopped** at step 13740 / 47556; not plotted |

Launch uses the frozen L3 token snapshot (`DS_DATA` pointing at that mix). Pack three
independent 8-GPU jobs (no 24-way shard). Small widths use FSDP
`NO_SHARD` (full replica per GPU). Do not refetch.

## Success gates

| Gate | Hit | Miss / stop |
|------|-----|-------------|
| L1 valley | C=3e18 BPB vs M over `{t0…t4,s1…s3}` has an **interior min** near t0–t2 | t0 BPB ~0.4, or monotone TinyStories collapse |
| No looping | `unique_train_tokens ≥ D` on every job | Point invalid |
| L2 move-right | Valley `M_opt` rises with C | Valley stuck or moves left |
| C3 fit | ≥2 budgets with on-grid valleys; `a ∈ [0.35, 0.70]` | Fit on endpoint minima |

## Launch

```bash
python3 scripts/make_grid.py --wave l1
python3 scripts/run_wave.py --grid configs/grid_wave_l.json

# Cluster packer (set DS_NODES first):
# python3 scripts/launch_v2.py --grid configs/grid_wave_l.json --max-jobs 3
# DS_GRID=configs/grid_wave_l.json bash scripts/install_watch_daemon.sh <head-host>

# L4: frozen 40B unique, 8-GPU pack (no 24-way shard)
python3 scripts/make_grid.py --wave l4
python3 scripts/run_wave.py --grid configs/grid_wave_l4.json
```
