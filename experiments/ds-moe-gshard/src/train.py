#!/usr/bin/env python3
"""DeepSeekMoE vs GShard trainer. 24-GPU FSDP NO_SHARD (no expert parallel)."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import PackedTokens  # noqa: E402
from lr import lr_at_frac  # noqa: E402
from moe import ffn_hidden  # noqa: E402
from model import MoELM  # noqa: E402

LN2 = math.log(2.0)


def smoke_lb_check(rec: dict) -> None:
    """Fail smoke if fixed balance loss is still negligible vs CE."""
    ce = float(rec.get("ce") or rec.get("loss") or 0.0)
    aux = float(rec.get("aux") or 0.0)
    max_frac = float(rec.get("max_expert_frac") or 1.0)
    aux_ce = aux / max(ce, 1e-9)
    if aux_ce < 0.005:
        raise SystemExit(
            f"smoke LB check failed: aux/CE={aux_ce:.6f} < 0.005 "
            f"(aux={aux:.6f} ce={ce:.4f}); balance loss likely still too weak"
        )
    if aux < 0.005:
        raise SystemExit(f"smoke LB check failed: aux={aux:.6f} < 0.005")
    print(
        f"smoke LB ok aux={aux:.4f} ce={ce:.4f} aux/CE={aux_ce:.4f} max_frac={max_frac:.3f}",
        flush=True,
    )
    if max_frac > 0.95:
        print(f"[warn] smoke max_frac={max_frac:.3f} still high; watch during E1", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--job-id", required=True)
    p.add_argument(
        "--arch-id",
        required=True,
        choices=("gshard", "gshard_x15", "dsmoe", "dsmoe_k3", "dsmoe_s0"),
    )
    p.add_argument("--n-layer", type=int, default=9)
    p.add_argument("--d-model", type=int, default=1280)
    p.add_argument("--n-heads", type=int, default=10)
    p.add_argument("--seq-len", type=int, default=2048)
    p.add_argument("--n-vocab", type=int, default=50257)
    p.add_argument("--d-tokens", type=float, default=10e9)
    p.add_argument("--max-lr", type=float, default=1.08e-3)
    p.add_argument("--batch-tokens", type=float, default=4e6)
    p.add_argument("--balance-alpha", type=float, default=0.01)
    p.add_argument("--init-std", type=float, default=0.006)
    p.add_argument("--micro-seqs", type=int, default=1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--train-bin", required=True)
    p.add_argument("--val-bin", required=True)
    p.add_argument("--train-bytes", default="")
    p.add_argument("--val-bytes", default="")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--eval-every-tokens", type=float, default=1e9)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--smoke-steps", type=int, default=0)
    p.add_argument("--no-checkpoint", action="store_true")
    p.add_argument("--resume", default="")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--ckpt-every-steps", type=int, default=50)
    return p.parse_args()


def load_bytes(path: str, n_tokens: int):
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
        amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        with amp:
            _, _loss = model(x, y)
        stats = getattr(_inner(model), "last_stats", {}) or {}
        nll = float(stats.get("ce", _loss.item()))
        ntok = x.numel()
        total_nll += nll * ntok
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


def dist_setup():
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


def _inner(model):
    m = model
    for attr in ("module", "_fsdp_wrapped_module"):
        if hasattr(m, attr):
            inner = getattr(m, attr)
            if inner is not None:
                m = inner
    return m


def save_ckpt(path: Path, model, opt, payload: dict, rank: int, world: int) -> None:
    import torch.distributed as dist

    path = Path(path)
    raw = _inner(model)
    model_sd = {k: v.detach().cpu() for k, v in raw.state_dict().items()}
    opt_sd = opt.state_dict()
    if rank == 0:
        tmp = path.with_suffix(".tmp")
        torch.save({"model": model_sd, "opt": opt_sd, **payload}, tmp)
        tmp.replace(path)
        print(f"[ckpt] wrote {path} step={payload.get('step')}", flush=True)
    if world > 1:
        dist.barrier()


def load_ckpt(path: Path, model, opt, rank: int) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    _inner(model).load_state_dict(ckpt["model"])
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

    micro = max(1, args.micro_seqs)
    tokens_per_micro = micro * args.seq_len * world
    accum = max(1, int(round(args.batch_tokens / tokens_per_micro)))
    tokens_per_step = accum * tokens_per_micro
    n_steps = max(1, int(math.ceil(args.d_tokens / tokens_per_step)))
    if args.smoke_steps > 0:
        n_steps = args.smoke_steps
    warmup_frac = min(2000 / n_steps, 0.08) if n_steps > 1 else 0.0
    eval_every = args.eval_every_tokens if args.eval_every_tokens > 0 else max(args.d_tokens / 10.0, tokens_per_step)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train.jsonl"

    train_c = Corpus(args.train_bin, args.train_bytes, args.seq_len, seed=args.seed + 1_000_003 * rank)
    val_c = Corpus(args.val_bin, args.val_bytes, args.seq_len, seed=args.seed + 7 + 1_000_003 * rank)

    model = MoELM(
        arch_id=args.arch_id,
        n_layer=args.n_layer,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_vocab=args.n_vocab,
        seq_len=args.seq_len,
        init_std=args.init_std,
        balance_alpha=args.balance_alpha,
    )
    model.gradient_checkpointing = not args.no_checkpoint and args.d_model >= 2048
    if args.no_checkpoint:
        model.gradient_checkpointing = False
    model = model.to(device)
    expert_params = int(model.expert_params())
    dense_ffn = int(model.dense_ffn_params())
    ffn0 = model.blocks[0].ffn
    one_exp = 3 * ffn0.routed.d_model * ffn0.routed.hidden
    act_slots = ffn0.n_shared + ffn0.k_routed
    activated_over_dense = (act_slots * one_exp * model.n_layer) / max(dense_ffn, 1)
    shard_name = "none"
    if world > 1:
        local = int(os.environ.get("LOCAL_RANK", rank))
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local],
            output_device=local,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )
        shard_name = "ddp"

    n_params = sum(p.numel() for p in model.parameters())

    try:
        opt = AdamW(model.parameters(), lr=args.max_lr, betas=(0.9, 0.95), weight_decay=0.1, fused=True)
    except TypeError:
        opt = AdamW(model.parameters(), lr=args.max_lr, betas=(0.9, 0.95), weight_decay=0.1)

    meta = {
        "job_id": args.job_id,
        "arch_id": args.arch_id,
        "n_layer": args.n_layer,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "seq_len": args.seq_len,
        "ffn_hidden_full": ffn_hidden(args.d_model),
        "D_tokens": args.d_tokens if args.smoke_steps == 0 else n_steps * tokens_per_step,
        "max_lr": args.max_lr,
        "batch_tokens_target": args.batch_tokens,
        "batch_tokens_actual": tokens_per_step,
        "micro_seqs": micro,
        "accum": accum,
        "n_steps": n_steps,
        "warmup_frac": warmup_frac,
        "n_params": int(n_params),
        "expert_params": expert_params,
        "dense_ffn_params": dense_ffn,
        "expert_over_dense": expert_params / max(dense_ffn, 1),
        "activated_over_dense": activated_over_dense,
        "k_routed": int(ffn0.k_routed),
        "n_shared": int(ffn0.n_shared),
        "world_size": world,
        "sharding": shard_name,
        "balance_alpha": args.balance_alpha,
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
    if is_main and train_c.tok.n < meta["D_tokens"]:
        print(
            f"[warn] unique_train_tokens={train_c.tok.n} < D_tokens={meta['D_tokens']}",
            flush=True,
        )
    if is_main:
        (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta, indent=2), flush=True)
        print(
            f"[arch] params={n_params/1e9:.3f}B experts={expert_params/1e9:.3f}B "
            f"({meta['expert_over_dense']:.3f}x dense FFN) "
            f"batch_actual={tokens_per_step} accum={accum}",
            flush=True,
        )

    interrupted = {"flag": False}

    def _on_term(signum, frame):
        interrupted["flag"] = True

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)

    ckpt_path = Path(args.resume) if args.resume else (out / "last.pt")
    do_resume = (not args.no_resume) and ckpt_path.exists()
    if do_resume and prev_world and prev_world != world:
        if is_main:
            print(f"[ckpt] skip resume: world_size={prev_world} != {world}", flush=True)
        do_resume = False
    step = 0
    tokens_seen = 0
    t0 = time.time()
    next_eval = eval_every
    best_bpb = None
    last_eval = None
    if do_resume:
        loaded = load_ckpt(ckpt_path, model, opt, rank)
        ckpt_world = loaded.get("world_size")
        if ckpt_world is not None and int(ckpt_world) != world:
            raise SystemExit(
                f"resume world_size={ckpt_world} != current {world} ({ckpt_path})"
            )
        step = int(loaded.get("step", 0))
        tokens_seen = int(loaded.get("tokens_seen", 0))
        best_bpb = loaded.get("best_bpb")
        last_eval = loaded.get("last_eval")
        next_eval = float(loaded.get("next_eval", tokens_seen + eval_every))
        if step >= n_steps:
            raise SystemExit(
                f"resume step={step} already >= n_steps={n_steps} for D={args.d_tokens}"
            )
        train_c.tok.rng = np.random.default_rng(args.seed + 1_000_003 * step + rank)
        if is_main:
            print(
                f"[ckpt] continue step={step}/{n_steps} tokens_seen={tokens_seen} "
                f"-> D={args.d_tokens}",
                flush=True,
            )
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
            "world_size": world,
            "job_id": args.job_id,
        }
        save_ckpt(out / "last.pt", model, opt, payload, rank, world)

    model.train()
    while step < n_steps:
        frac = step / n_steps
        lr = lr_at_frac(frac, args.max_lr, warmup_frac)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        ce_acc = 0.0
        aux_acc = 0.0
        frac_acc = 0.0
        t_step = time.time()
        amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        for mi in range(accum):
            x, y, _ = train_c.batch(micro, device)
            sync_ctx = (
                contextlib.nullcontext()
                if world == 1 or mi == accum - 1
                else model.no_sync()
            )
            with sync_ctx, amp:
                _, loss = model(x, y)
            (loss / accum).backward()
            loss_acc += float(loss.item())
            stats = getattr(_inner(model), "last_stats", {}) or {}
            ce_acc += float(stats.get("ce", 0.0))
            aux_acc += float(stats.get("aux", 0.0))
            frac_acc = max(frac_acc, float(stats.get("max_expert_frac", 0.0)))
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
            "ce": ce_acc / accum,
            "aux": aux_acc / accum,
            "aux_ce_ratio": (aux_acc / accum) / max(ce_acc / accum, 1e-9),
            "max_expert_frac": frac_acc,
            "lr": lr,
            "tok_s": tps,
        }
        for key in (
            "routing_layer",
            "load_imbalance",
            "token_frac_entropy",
            "gate_entropy",
            "token_frac_top",
            "gate_mean_top",
            "layer_max_frac",
        ):
            val = stats.get(key)
            if val is not None:
                rec[key] = val
        if is_main and (step % args.log_every == 0 or step == 1):
            print(
                f"step {step}/{n_steps} loss={rec['loss']:.4f} ce={rec['ce']:.4f} "
                f"aux={rec['aux']:.4f} max_frac={frac_acc:.3f} lr={lr:.3e} "
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
        if interrupted["flag"] or (
            args.smoke_steps == 0 and (step % args.ckpt_every_steps == 0 or step == n_steps)
        ):
            dump_ckpt()
        if interrupted["flag"]:
            if is_main:
                print(f"[ckpt] SIGTERM at step={step}; exiting for resume", flush=True)
            break

    if logf:
        logf.close()
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
            "arch_id": args.arch_id,
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
        if args.smoke_steps > 0:
            smoke_lb_check(rec)
            print("SMOKE_OK", args.job_id, flush=True)
    if world > 1:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
