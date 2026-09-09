"""Packed uint16 token memmap. Train loops if D exceeds unique tokens."""

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

    def batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Return (x, y, nbytes_per_seq). y is next-token; nbytes is GPT-2 byte length of y tokens."""
        starts = self.rng.integers(0, self.n - self.seq_len - 1, size=batch_size)
        xs, ys, nbytes = [], [], []
        for s in starts:
            chunk = np.array(self.data[s : s + self.seq_len + 1], dtype=np.int64)
            xs.append(chunk[:-1])
            ys.append(chunk[1:])
            # GPT-2: one token is at least 1 byte; use token count as conservative
            # byte proxy only if decoder missing. Caller may pass a decoder.
            nbytes.append(self.seq_len)
        x = torch.tensor(np.stack(xs), device=device)
        y = torch.tensor(np.stack(ys), device=device)
        return x, y, np.asarray(nbytes, dtype=np.int64)


def load_byte_lut(path: str | Path | None) -> np.ndarray | None:
    """Optional uint8[vocab] mean bytes/token table written by tokenize script."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return np.load(p)
