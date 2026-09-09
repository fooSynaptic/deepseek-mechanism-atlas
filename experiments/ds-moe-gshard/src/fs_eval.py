#!/usr/bin/env python3
"""Eval-only F/S ablations on a trained DeepSeekMoE checkpoint (paper §4.5).

Loads once, sweeps:
  F: k_routed in {3,4,5,6,7} with shared on
  S: n_shared=0, k_routed=8
P0 gate: baseline k=7 + shared must be near E1 val BPB.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from train import Corpus, evaluate, dist_setup, _inner  # noqa: E402
from model import MoELM  # noqa: E402

LN2 = math.log(2.0)
E1_DSMOE_BPB = 0.7850027091184383
GSHARD_BPB = 0.797006921333703
BASELINE_TOL = 0.05


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--job-id", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--val-bin", required=True)
    p.add_argument("--val-bytes", default="")
    p.add_argument("--n-layer", type=int, default=9)
    p.add_argument("--d-model", type=int, default=1280)
    p.add_argument("--n-heads", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--n-vocab", type=int, default=50257)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--micro-seqs", type=int, default=1)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--balance-alpha", type=float, default=0.01)
    p.add_argument("--baseline-bpb", type=float, default=E1_DSMOE_BPB)
    p.add_argument("--baseline-tol", type=float, default=BASELINE_TOL)
    p.add_argument("--skip-gate", action="store_true")
    p.add_argument(
        "--gshard-ckpt",
        default="",
        help="Optional GShard ckpt evaluated on the same val batches (fair F ref)",
    )
    return p.parse_args()


def load_model_weights(path: Path, model: torch.nn.Module, rank: int) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    _inner(model).load_state_dict(sd)
    if rank == 0:
        print(
            f"[fs_eval] loaded {path} step={ckpt.get('step') if isinstance(ckpt, dict) else '?'}",
            flush=True,
        )
    return ckpt if isinstance(ckpt, dict) else {}


def reset_val_rng(corpus: Corpus, seed: int, rank: int) -> None:
    corpus.tok.rng = np.random.default_rng(seed + 7 + 1_000_003 * rank)


def run_one(
    model,
    corpus: Corpus,
    *,
    k_routed: int,
    n_shared: int,
    batches: int,
    micro: int,
    device,
    seed: int,
    rank: int,
) -> dict:
    reset_val_rng(corpus, seed, rank)
    _inner(model).set_routing_override(k_routed=k_routed, n_shared=n_shared)
    ev = evaluate(model, corpus, batches, micro, device)
    return {
        "k_routed": k_routed,
        "n_shared": n_shared,
        "activated_quarter": n_shared + k_routed,
        **ev,
    }


def main() -> None:
    args = parse_args()
    rank, world, device = dist_setup()
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if device.type != "cuda":
        raise SystemExit("CUDA required")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    val_c = Corpus(args.val_bin, args.val_bytes, args.seq_len, seed=args.seed + 7)

    model = MoELM(
        arch_id="dsmoe",
        n_layer=args.n_layer,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_vocab=args.n_vocab,
        seq_len=args.seq_len,
        balance_alpha=args.balance_alpha,
    ).to(device)

    ckpt_meta = load_model_weights(Path(args.ckpt), model, rank)
    if world > 1:
        local = int(os.environ.get("LOCAL_RANK", rank))
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local],
            output_device=local,
            find_unused_parameters=False,
        )

    micro = max(1, args.micro_seqs)
    t0 = time.time()
    configs: list[tuple[str, int, int]] = [
        ("p0_baseline", 7, 1),
        ("f_k3", 3, 1),
        ("f_k4", 4, 1),
        ("f_k5", 5, 1),
        ("f_k6", 6, 1),
        ("f_k7", 7, 1),
        ("s_noshare_k8", 8, 0),
    ]
    rows: list[dict] = []
    for name, k, ns in configs:
        if is_main:
            print(f"[fs_eval] {name} k={k} n_shared={ns} …", flush=True)
        row = run_one(
            model,
            val_c,
            k_routed=k,
            n_shared=ns,
            batches=args.eval_batches,
            micro=micro,
            device=device,
            seed=args.seed,
            rank=rank,
        )
        row["name"] = name
        rows.append(row)
        if is_main:
            print(
                f"[fs_eval] {name} bpb={row['val_bpb']:.4f} nll={row['val_nll']:.4f}",
                flush=True,
            )
        if world > 1:
            import torch.distributed as dist

            dist.barrier()

    baseline = next(r for r in rows if r["name"] == "p0_baseline")
    gate_ok = abs(float(baseline["val_bpb"]) - args.baseline_bpb) <= args.baseline_tol
    if not args.skip_gate and not gate_ok:
        msg = (
            f"P0 gate FAIL: baseline bpb={baseline['val_bpb']:.4f} "
            f"vs E1 {args.baseline_bpb:.4f} (tol={args.baseline_tol})"
        )
        if is_main:
            payload = {
                "job_id": args.job_id,
                "ok": False,
                "gate_ok": False,
                "error": msg,
                "rows": rows,
                "wall_s": time.time() - t0,
                "ckpt_step": ckpt_meta.get("step"),
                "eval_batches": args.eval_batches,
                "world_size": world,
                "gshard_ref_bpb": GSHARD_BPB,
                "e1_dsmoe_ref_bpb": args.baseline_bpb,
            }
            (out / "result.json").write_text(json.dumps(payload, indent=2) + "\n")
            print(msg, flush=True)
        if world > 1:
            import torch.distributed as dist

            dist.barrier()
            dist.destroy_process_group()
        raise SystemExit(msg)

    f_rows = [r for r in rows if r["name"].startswith("f_")]
    s_row = next(r for r in rows if r["name"] == "s_noshare_k8")
    gshard_row = None
    if args.gshard_ckpt:
        if is_main:
            print(f"[fs_eval] loading gshard ref {args.gshard_ckpt}", flush=True)
        if world > 1:
            import torch.distributed as dist

            dist.barrier()
        del model
        torch.cuda.empty_cache()
        gmodel = MoELM(
            arch_id="gshard",
            n_layer=args.n_layer,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_vocab=args.n_vocab,
            seq_len=args.seq_len,
            balance_alpha=args.balance_alpha,
        ).to(device)
        load_model_weights(Path(args.gshard_ckpt), gmodel, rank)
        if world > 1:
            local = int(os.environ.get("LOCAL_RANK", rank))
            gmodel = torch.nn.parallel.DistributedDataParallel(
                gmodel,
                device_ids=[local],
                output_device=local,
                find_unused_parameters=False,
            )
        reset_val_rng(val_c, args.seed, rank)
        gev = evaluate(gmodel, val_c, args.eval_batches, micro, device)
        gshard_row = {"name": "gshard_ref", "k_routed": 2, "n_shared": 0, **gev}
        if is_main:
            print(
                f"[fs_eval] gshard_ref bpb={gshard_row['val_bpb']:.4f} "
                f"nll={gshard_row['val_nll']:.4f}",
                flush=True,
            )
        del gmodel
        torch.cuda.empty_cache()

    if is_main:
        result = {
            "job_id": args.job_id,
            "ok": True,
            "gate_ok": gate_ok,
            "wall_s": time.time() - t0,
            "ckpt": str(args.ckpt),
            "ckpt_step": ckpt_meta.get("step"),
            "eval_batches": args.eval_batches,
            "world_size": world,
            "gshard_ref_bpb_e1": GSHARD_BPB,
            "e1_dsmoe_ref_bpb": args.baseline_bpb,
            "baseline": baseline,
            "baseline_minus_e1": float(baseline["val_bpb"]) - args.baseline_bpb,
            "f_eval": f_rows,
            "s_eval": s_row,
            "delta_s_vs_baseline": float(s_row["val_bpb"]) - float(baseline["val_bpb"]),
            "gshard_same_batches": gshard_row,
            "rows": rows,
        }
        if gshard_row is not None:
            result["f_vs_gshard"] = [
                {
                    "name": r["name"],
                    "k_routed": r["k_routed"],
                    "val_bpb": r["val_bpb"],
                    "delta_vs_gshard": float(r["val_bpb"]) - float(gshard_row["val_bpb"]),
                }
                for r in f_rows
            ]
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        (out / "meta.json").write_text(
            json.dumps(
                {
                    "job_id": args.job_id,
                    "protocol": "fs_eval_p0_f_s",
                    "ckpt": str(args.ckpt),
                    "gshard_ckpt": args.gshard_ckpt or None,
                    "eval_batches": args.eval_batches,
                    "seed": args.seed,
                    "world_size": world,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            "DONE",
            json.dumps(
                {
                    "job_id": args.job_id,
                    "baseline_bpb": baseline["val_bpb"],
                    "s_delta": result["delta_s_vs_baseline"],
                    "gshard_same_bpb": None if gshard_row is None else gshard_row["val_bpb"],
                    "wall_s": result["wall_s"],
                }
            ),
            flush=True,
        )
        print("FS_EVAL_OK", args.job_id, flush=True)

    if world > 1:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
