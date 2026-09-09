#!/usr/bin/env python3
"""Print 2B skeleton expert-param lock. GShard vs DeepSeekMoE must match to 1%."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def ffn_hidden(d_model: int) -> int:
    raw = (8.0 / 3.0) * d_model
    return max(64, int(round(raw / 64.0) * 64))


def round_hidden(h: float) -> int:
    return max(64, int(round(h / 64.0) * 64))


def formula(d_model: int = 1280, n_layer: int = 9) -> dict:
    h = ffn_hidden(d_model)
    h15 = round_hidden(h * 1.5)
    one_full = 3 * d_model * h
    one_x15 = 3 * d_model * h15
    dense = one_full * n_layer
    gshard = 16 * one_full * n_layer
    gshard_x15 = 16 * one_x15 * n_layer
    dsmoe = 64 * 3 * d_model * (h // 4) * n_layer
    # F-train: same total params as dsmoe; activate 1 shared + Top-3 = 4/64 of pool
    dsmoe_k3_activated = dsmoe * 4 / 64
    return {
        "ffn_hidden_full": h,
        "ffn_hidden_x15": h15,
        "ffn_hidden_quarter": h // 4,
        "dense_ffn_params": dense,
        "gshard_expert_params": gshard,
        "gshard_x15_expert_params": gshard_x15,
        "dsmoe_expert_params": dsmoe,
        "gshard_activated": gshard * 2 / 16,
        "gshard_x15_activated": gshard_x15 * 2 / 16,
        "dsmoe_activated": dsmoe * 8 / 64,
        "dsmoe_k3_activated": dsmoe_k3_activated,
        # S-train: 64 routed, Top-8, no shared — same totals as dsmoe
        "dsmoe_s0_expert_params": dsmoe,
        "dsmoe_s0_activated": dsmoe * 8 / 64,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--instantiate",
        action="store_true",
        help="Build both models (needs ~8GB RAM each). Default is formula-only.",
    )
    args = ap.parse_args()
    row = formula()
    rel_e = abs(row["gshard_expert_params"] - row["dsmoe_expert_params"]) / row["gshard_expert_params"]
    rel_a = abs(row["gshard_activated"] - row["dsmoe_activated"]) / row["gshard_activated"]
    print(
        f"h_full={row['ffn_hidden_full']} h_x15={row['ffn_hidden_x15']} "
        f"h_quarter={row['ffn_hidden_quarter']} dense={row['dense_ffn_params']/1e6:.1f}M"
    )
    print(
        f"gshard experts={row['gshard_expert_params']/1e9:.4f}B "
        f"activated={row['gshard_activated']/1e9:.4f}B "
        f"({row['gshard_expert_params']/row['dense_ffn_params']:.3f}x dense)"
    )
    print(
        f"gshard_x15 experts={row['gshard_x15_expert_params']/1e9:.4f}B "
        f"activated={row['gshard_x15_activated']/1e9:.4f}B "
        f"({row['gshard_x15_expert_params']/row['dense_ffn_params']:.3f}x dense) "
        f"vs_gshard={row['gshard_x15_expert_params']/row['gshard_expert_params']:.3f}x"
    )
    print(
        f"dsmoe  experts={row['dsmoe_expert_params']/1e9:.4f}B "
        f"activated={row['dsmoe_activated']/1e9:.4f}B "
        f"({row['dsmoe_expert_params']/row['dense_ffn_params']:.3f}x dense)"
    )
    print(
        f"dsmoe_k3 (F-train) same experts; "
        f"activated={row['dsmoe_k3_activated']/1e9:.4f}B "
        f"({row['dsmoe_k3_activated']/row['dense_ffn_params']:.3f}x dense) "
        f"vs_gshard_act={row['dsmoe_k3_activated']/row['gshard_activated']:.3f}x"
    )
    print(f"expert_param_rel_diff={rel_e:.6f} activated_rel_diff={rel_a:.6f}")
    if rel_e > 0.01 or rel_a > 0.01:
        raise SystemExit("C1 miss: expert or activated params differ by >1%")
    print("C1 formula lock OK (gshard vs dsmoe)")
    half = abs(row["dsmoe_k3_activated"] / row["gshard_activated"] - 0.5)
    if half > 0.01:
        raise SystemExit("F-train lock miss: dsmoe_k3 activated should be ~0.5× GShard")
    print("F-train formula lock OK (dsmoe_k3 activated ≈ 0.5× GShard)")
    rel_s0_e = abs(row["dsmoe_s0_expert_params"] - row["dsmoe_expert_params"]) / row["dsmoe_expert_params"]
    rel_s0_a = abs(row["dsmoe_s0_activated"] - row["dsmoe_activated"]) / row["dsmoe_activated"]
    print(
        f"dsmoe_s0 (S-train) experts={row['dsmoe_s0_expert_params']/1e9:.4f}B "
        f"activated={row['dsmoe_s0_activated']/1e9:.4f}B "
        f"(same lock as dsmoe; rel_e={rel_s0_e:.6f} rel_a={rel_s0_a:.6f})"
    )
    if rel_s0_e > 0.01 or rel_s0_a > 0.01:
        raise SystemExit("S-train lock miss: dsmoe_s0 params/activated must match dsmoe")
    print("S-train formula lock OK (dsmoe_s0 ≡ dsmoe totals)")

    if not args.instantiate:
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from model import MoELM  # noqa: E402

    for arch in ("gshard", "gshard_x15", "dsmoe", "dsmoe_k3", "dsmoe_s0"):
        m = MoELM(arch, n_layer=9, d_model=1280, n_heads=10, n_vocab=50257, seq_len=2048)
        exp = m.expert_params()
        dense = m.dense_ffn_params()
        print(
            f"{arch:12s} n_params={sum(p.numel() for p in m.parameters())/1e9:.3f}B "
            f"experts={exp/1e9:.3f}B ({exp/dense:.3f}x) h={m.blocks[0].ffn.hidden}"
        )
        del m
    print("instantiate lock OK")

if __name__ == "__main__":
    main()
