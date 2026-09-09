#!/usr/bin/env python3
"""IsoFLOP / Formula-1 trainer (DeepSeek-LLM V1). 1 GPU or FSDP via torchrun."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import PackedTokens  # noqa: E402
from flops import formula1_batch_tokens, formula1_lr, make_arch, tokens_for_compute  # noqa: E402
from lr import lr_at_frac  # noqa: E402
from model import V1DenseLM  # noqa: E402

LN2 = math.log(2.0)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--job-id", required=True)
    p.add_argument("--arch-id", required=True)
    p.add_argument("--c-flops", type=float, required=True)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--n-vocab", type=int, default=50257)
    p.add_argument("--checkpoint", action="store_true")
    p.add_argument(
        "--no-checkpoint",
        action="store_true",
        help="force gradient checkpointing off (recompute costs ~30% when VRAM is free)",
    )
    p.add_argument("--train-bin", type=str, required=True)
    p.add_argument("--val-bin", type=str, required=True)
    p.add_argument("--train-bytes", type=str, default="")
    p.add_argument("--val-bytes", type=str, default="")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--batch-tokens", type=float, default=0.0, help="0 = Formula 1 B_opt")
    p.add_argument("--max-lr", type=float, default=0.0, help="0 = Formula 1 eta_opt")
    p.add_argument("--micro-seqs", type=int, default=0, help="0 = auto")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eval-every-tokens", type=float, default=0.0)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--smoke-steps", type=int, default=0)
    p.add_argument("--init-std", type=float, default=0.006)
    p.add_argument("--resume", type=str, default="", help="ckpt path; default out-dir/last.pt if present")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--ckpt-every-steps", type=int, default=50)
    return p.parse_args()


def load_bytes(path: str, n_tokens: int) -> np.ndarray | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    arr = np.memmap(p, dtype=np.uint8, mode="r")
    if len(arr) != n_tokens:
        print(f"[warn] bytes length {len(arr)} != tokens {n_tokens}; ignoring", flush=True)
        return None
    return arr


class Corpus:
    def __init__(self, tok_path: str, bytes_path: str, seq_len: int, seed: int):
        self.tok = PackedTokens(tok_path, seq_len, seed=seed)
        self.nbytes = load_bytes(bytes_path, self.tok.n)

    def batch(self, bs: int, device: torch.device):
        starts = self.tok.rng.integers(0, self.tok.n - self.tok.seq_len - 1, size=bs)
        xs, ys, nb = [], [], []
        sl = self.tok.seq_len
        for s in starts:
            chunk = np.asarray(self.tok.data[s : s + sl + 1], dtype=np.int64)
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
            if self.nbytes is not None:
                nb.append(int(np.asarray(self.nbytes[s + 1 : s + sl + 1]).sum()))
            else:
                nb.append(sl)
        x = torch.tensor(np.stack(xs), device=device)
        y = torch.tensor(np.stack(ys), device=device)
        return x, y, float(sum(nb))


@torch.no_grad()
def evaluate(model, corpus: Corpus, batches: int, micro: int, device) -> dict:
    model.eval()
    total_nll = 0.0
    total_tok = 0
    total_bytes = 0.0
    for _ in range(batches):
        x, y, nbytes = corpus.batch(micro, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)  # eval: autocast is fine under FSDP too
        ntok = x.numel()
        total_nll += float(loss.item()) * ntok
        total_tok += ntok
        total_bytes += nbytes
    model.train()
    mean_nll = total_nll / max(total_tok, 1)
    bpb = total_nll / max(total_bytes * LN2, 1e-9)
    return {
        "val_nll": mean_nll,
        "val_bpb": bpb,
        "val_tokens": total_tok,
        "val_bytes": total_bytes,
    }


def auto_micro_seqs(d_model: int, seq_len: int) -> int:
    if d_model >= 3072:
        return 1
    if d_model >= 2048:
        return max(1, 8192 // seq_len)
    if d_model >= 1280:
        return max(1, 16384 // seq_len)
    return max(2, 32768 // seq_len)


def dist_setup():
    """Return (global_rank, world_size, device). Device uses LOCAL_RANK."""
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws <= 1:
        return 0, 1, torch.device("cuda")
    import datetime
    import torch.distributed as dist

    print(f"[dist] rank wait LOCAL_RANK={os.environ.get('LOCAL_RANK')} WORLD_SIZE={ws}", flush=True)
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=10))
    print(f"[dist] init ok rank={dist.get_rank()} world={ws} local={local}", flush=True)
    return dist.get_rank(), ws, torch.device(f"cuda:{local}")


def _is_fsdp(model) -> bool:
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return isinstance(model, FSDP)
    except Exception:
        return False


def save_ckpt(path: Path, model, opt, payload: dict, rank: int, world: int) -> None:
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig, StateDictType

    path = Path(path)
    if _is_fsdp(model):
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        ocfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg, ocfg):
            model_sd = model.state_dict()
            opt_sd = FSDP.optim_state_dict(model, opt)
    else:
        model_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        opt_sd = opt.state_dict()
    if rank == 0:
        tmp = path.with_suffix(".tmp")
        torch.save({"model": model_sd, "opt": opt_sd, **payload}, tmp)
        tmp.replace(path)
        print(f"[ckpt] wrote {path} step={payload.get('step')}", flush=True)
    if world > 1:
        dist.barrier()


def load_ckpt(path: Path, model, opt, rank: int) -> dict:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig, StateDictType

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if _is_fsdp(model):
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
        ocfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg, ocfg):
            model.load_state_dict(ckpt["model"])
            opt_sd = FSDP.optim_state_dict_to_load(model, opt, ckpt["opt"])
            opt.load_state_dict(opt_sd)
    else:
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
    if rank == 0:
        print(f"[ckpt] resumed {path} step={ckpt.get('step')}", flush=True)
    return ckpt


def main():
    args = parse_args()
    rank, world, device = dist_setup()
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    if device.type != "cuda":
        raise SystemExit("CUDA required")

    arch = make_arch(args.arch_id, seq_len=args.seq_len, n_vocab=args.n_vocab)
    c = float(args.c_flops)
    d_tokens = tokens_for_compute(c, arch.M)
    max_lr = args.max_lr if args.max_lr > 0 else formula1_lr(c)
    batch_tokens = args.batch_tokens if args.batch_tokens > 0 else formula1_batch_tokens(c)
    micro = args.micro_seqs if args.micro_seqs > 0 else auto_micro_seqs(arch.d_model, arch.seq_len)
    tokens_per_micro = micro * arch.seq_len * world
    accum = max(1, int(round(batch_tokens / tokens_per_micro)))
    tokens_per_step = accum * tokens_per_micro
    n_steps = max(1, int(math.ceil(d_tokens / tokens_per_step)))
    if args.smoke_steps > 0:
        n_steps = args.smoke_steps
        d_tokens = n_steps * tokens_per_step
    warmup_frac = min(2000 / n_steps, 0.05) if n_steps > 1 else 0.0
    eval_every = args.eval_every_tokens if args.eval_every_tokens > 0 else max(d_tokens / 10.0, tokens_per_step)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.jsonl"

    train_c = Corpus(args.train_bin, args.train_bytes, arch.seq_len, seed=args.seed + 1_000_003 * rank)
    val_c = Corpus(args.val_bin, args.val_bytes, arch.seq_len, seed=args.seed + 7 + 1_000_003 * rank)

    ckpt = False if args.no_checkpoint else bool(args.checkpoint or arch.d_model >= 2048)
    model = V1DenseLM(
        n_layer=arch.n_layer,
        d_model=arch.d_model,
        n_heads=arch.n_heads,
        n_vocab=arch.n_vocab,
        seq_len=arch.seq_len,
        init_std=args.init_std,
    )
    model.gradient_checkpointing = ckpt
    model = model.to(device)
    shard_name = "none"
    if world > 1:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from functools import partial
        from model import Block

        mp = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
        # Small IsoFLOP widths (u*/t*) fit on one GPU. FULL_SHARD across 24 ranks
        # is not TP; it just slices a 30M model to ~3% VRAM and adds all-gather.
        # Keep a full replica (NO_SHARD ≈ DDP) unless the 7B-class widths need it.
        shard = (
            ShardingStrategy.FULL_SHARD
            if arch.d_model >= 2048
            else ShardingStrategy.NO_SHARD
        )
        shard_name = "full_shard" if shard == ShardingStrategy.FULL_SHARD else "no_shard"
        model = FSDP(
            model,
            auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={Block}),
            mixed_precision=mp,
            sharding_strategy=shard,
            device_id=int(os.environ.get("LOCAL_RANK", rank)),
            use_orig_params=True,
        )
    n_params = int(arch.N1 + arch.n_vocab * arch.d_model)

    try:
        opt = AdamW(
            model.parameters(),
            lr=max_lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=True,
        )
    except TypeError:
        opt = AdamW(
            model.parameters(),
            lr=max_lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )

    meta = {
        "job_id": args.job_id,
        "arch": arch.scale_table(),
        "c_flops": c,
        "D_tokens": d_tokens,
        "max_lr": max_lr,
        "batch_tokens_target": batch_tokens,
        "batch_tokens_actual": tokens_per_step,
        "micro_seqs": micro,
        "accum": accum,
        "n_steps": n_steps,
        "warmup_frac": warmup_frac,
        "n_params": n_params,
        "world_size": world,
        "sharding": shard_name,
        "checkpoint": ckpt,
        "device": torch.cuda.get_device_name(0),
        "unique_train_tokens": train_c.tok.n,
        "unique_val_tokens": val_c.tok.n,
    }
    prev_world = None
    prev_meta = out / "meta.json"
    if prev_meta.exists():
        try:
            prev_world = int(json.loads(prev_meta.read_text()).get("world_size") or 0)
        except Exception:
            prev_world = None
    if is_main and train_c.tok.n < d_tokens:
        print(
            f"[warn] unique_train_tokens={train_c.tok.n} < D_tokens={d_tokens}; "
            "this job will loop the corpus",
            flush=True,
        )
    if is_main:
        (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta, indent=2), flush=True)

    interrupted = {"flag": False}

    def _on_term(signum, frame):
        interrupted["flag"] = True

    import signal

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    ckpt_path = Path(args.resume) if args.resume else (out / "last.pt")
    do_resume = (not args.no_resume) and ckpt_path.exists()
    if do_resume and prev_world and prev_world != world:
        if is_main:
            print(
                f"[ckpt] skip resume: last.pt world_size={prev_world} != {world}",
                flush=True,
            )
        do_resume = False
    step = 0
    tokens_seen = 0
    t0 = time.time()
    next_eval = eval_every
    best_bpb = None
    last_eval = None
    if do_resume:
        loaded = load_ckpt(ckpt_path, model, opt, rank)
        step = int(loaded.get("step", 0))
        tokens_seen = int(loaded.get("tokens_seen", 0))
        best_bpb = loaded.get("best_bpb")
        last_eval = loaded.get("last_eval")
        next_eval = float(loaded.get("next_eval", tokens_seen + eval_every))
        train_c.tok.rng = np.random.default_rng(args.seed + 1_000_003 * step + rank)
        if world > 1:
            import torch.distributed as dist

            dist.barrier()
    logf = None
    if is_main:
        logf = log_path.open("a" if do_resume else "w")

    rec = {"loss": None}

    def dump_ckpt():
        payload = {
            "step": step,
            "tokens_seen": tokens_seen,
            "best_bpb": best_bpb,
            "last_eval": last_eval,
            "next_eval": next_eval,
            "numpy_rng": train_c.tok.rng.bit_generator.state,
            "world_size": world,
            "job_id": args.job_id,
        }
        save_ckpt(out / "last.pt", model, opt, payload, rank, world)

    model.train()

    while step < n_steps:
        frac = step / n_steps
        lr = lr_at_frac(frac, max_lr, warmup_frac)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        t_step = time.time()
        amp = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if world == 1
            else contextlib.nullcontext()
        )
        for _ in range(accum):
            x, y, _ = train_c.batch(micro, device)
            with amp:
                _, loss = model(x, y)
            (loss / accum).backward()
            loss_acc += float(loss.item())
        if world > 1:
            model.clip_grad_norm_(1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        tokens_seen += tokens_per_step
        dt = time.time() - t_step
        tps = tokens_per_step / max(dt, 1e-6)
        rec = {
            "step": step,
            "tokens": tokens_seen,
            "loss": loss_acc / accum,
            "lr": lr,
            "tok_s": tps,
        }
        if is_main and (step % args.log_every == 0 or step == 1):
            print(
                f"step {step}/{n_steps} loss={rec['loss']:.4f} lr={lr:.3e} "
                f"tok/s={tps:.0f} seen={tokens_seen}",
                flush=True,
            )
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
        if tokens_seen >= next_eval or step == n_steps:
            ev = evaluate(model, val_c, args.eval_batches, micro, device)
            last_eval = ev
            if best_bpb is None or ev["val_bpb"] < best_bpb:
                best_bpb = ev["val_bpb"]
            if is_main:
                print(
                    f"eval step={step} nll={ev['val_nll']:.4f} bpb={ev['val_bpb']:.4f}",
                    flush=True,
                )
                logf.write(json.dumps({"step": step, "tokens": tokens_seen, **ev}) + "\n")
                logf.flush()
            next_eval += eval_every
        if world > 1:
            import torch.distributed as dist

            flag = torch.tensor([1 if interrupted["flag"] else 0], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            interrupted["flag"] = bool(flag.item())
        if interrupted["flag"] or step % args.ckpt_every_steps == 0 or step == n_steps:
            dump_ckpt()
        if interrupted["flag"]:
            if is_main:
                print(f"[ckpt] SIGTERM at step={step}; exiting for resume", flush=True)
            break

    if logf:
        logf.close()
        logf = None
    if interrupted["flag"]:
        if world > 1:
            import torch.distributed as dist

            dist.barrier()
            dist.destroy_process_group()
        return
    if is_main:
        result = {
            "job_id": args.job_id,
            "ok": True,
            "steps": step,
            "tokens_seen": tokens_seen,
            "wall_s": time.time() - t0,
            "final_train_loss": rec["loss"],
            "final_eval": last_eval,
            "best_val_bpb": best_bpb,
            "meta": meta,
        }
        (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print("DONE", json.dumps({k: result[k] for k in ("job_id", "best_val_bpb", "wall_s")}), flush=True)
    if world > 1:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
