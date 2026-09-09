"""Packed uint16 token memmap."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class PackedTokens:
    def __init__(self, path: str | Path, seq_len: int, seed: int = 1):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        if len(self.data) < seq_len + 1:
            raise ValueError(f"{path} too short: {len(self.data)} tokens")
        self.seq_len = seq_len
        self.n = len(self.data)
        self.rng = np.random.default_rng(seed)
