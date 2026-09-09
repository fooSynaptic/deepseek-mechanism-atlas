#!/usr/bin/env python3
"""Write a tiny random packed corpus so smoke does not wait on HF."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-tokens", type=int, default=2_000_000)
    p.add_argument("--val-tokens", type=int, default=200_000)
    p.add_argument("--vocab", type=int, default=50257)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tr = rng.integers(0, args.vocab, size=args.train_tokens, dtype=np.uint16)
    va = rng.integers(0, args.vocab, size=args.val_tokens, dtype=np.uint16)
    tr.tofile(out / "train.bin")
    va.tofile(out / "val.bin")
    np.full(args.train_tokens, 3, dtype=np.uint8).tofile(out / "train.bytes.bin")
    np.full(args.val_tokens, 3, dtype=np.uint8).tofile(out / "val.bytes.bin")
    print(f"smoke data {out} train={args.train_tokens} val={args.val_tokens}")


if __name__ == "__main__":
    main()
