"""DeepSeek-LLM V1 Formula 1 / 2 helpers.

Paper: arXiv:2401.02954 §3. M is non-embedding FLOPs/token (forward).
Training compute uses C = M * D (tokens). Backward is not folded into M.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


def n1_params(n_layer: int, d_model: int) -> float:
    """Non-embedding parameter count implied by Formula 2 (N1 = 12 n d^2)."""
    return 12.0 * n_layer * (d_model**2)


def six_n1(n_layer: int, d_model: int) -> float:
    return 72.0 * n_layer * (d_model**2)


def six_n2(n_layer: int, d_model: int, n_vocab: int) -> float:
    return six_n1(n_layer, d_model) + 6.0 * n_vocab * d_model


def m_flops_per_token(n_layer: int, d_model: int, seq_len: int) -> float:
    """Formula 2: M = 72 n d^2 + 12 n d L."""
    return six_n1(n_layer, d_model) + 12.0 * n_layer * d_model * seq_len


def formula1_lr(c_flops: float) -> float:
    return 0.3118 * (c_flops ** -0.1250)


def formula1_batch_tokens(c_flops: float) -> float:
    return 0.2920 * (c_flops ** 0.3271)


def ffn_hidden(d_model: int) -> int:
    """SwiGLU inner dim: 8/3 d, rounded to multiple of 64."""
    raw = (8.0 / 3.0) * d_model
    return max(64, int(round(raw / 64.0) * 64))


@dataclass(frozen=True)
class Arch:
    arch_id: str
    n_layer: int
    d_model: int
    n_heads: int
    seq_len: int
    n_vocab: int

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def ffn(self) -> int:
        return ffn_hidden(self.d_model)

    @property
    def M(self) -> float:
        return m_flops_per_token(self.n_layer, self.d_model, self.seq_len)

    @property
    def N1(self) -> float:
        return n1_params(self.n_layer, self.d_model)

    @property
    def six_N1(self) -> float:
        return six_n1(self.n_layer, self.d_model)

    @property
    def six_N2(self) -> float:
        return six_n2(self.n_layer, self.d_model, self.n_vocab)

    def scale_table(self) -> dict:
        m = self.M
        return {
            "arch_id": self.arch_id,
            "n_layer": self.n_layer,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "seq_len": self.seq_len,
            "n_vocab": self.n_vocab,
            "ffn_hidden": self.ffn,
            "N1": self.N1,
            "M": m,
            "six_N1": self.six_N1,
            "six_N2": self.six_N2,
            "six_N1_over_M": self.six_N1 / m,
            "six_N2_over_M": self.six_N2 / m,
        }


def tokens_for_compute(c_flops: float, m: float) -> int:
    return int(math.ceil(c_flops / m))


# IsoFLOP widths at V1 product scale (head_dim = 128 for ≥2k d, else 64).
# s5 matches DeepSeek-LLM 7B Table 2: 30L / d=4096 / 32 heads / MHA.
# All IsoFLOP jobs: single-node 8-GPU FSDP (world_size=8). Three nodes may
# each run one independent job; no multi-node torchrun.
ARCHS: dict[str, tuple[int, int, int]] = {
    "s1": (22, 2048, 16),   # ~1.11B N1
    "s2": (24, 2560, 20),   # ~1.89B
    "s3": (26, 3072, 24),   # ~2.94B
    "s4": (28, 3584, 28),   # ~4.32B
    "s5": (30, 4096, 32),   # ~6.04B  == V1 7B
    "s6": (32, 4096, 32),   # ~6.44B  Table 3 32×4096
    # Wave L: left of s1, aligned to paper M_opt at C∈{3e18,1e19,3e19}
    "t0": (12, 768, 6),     # M~9.63e8  near M_opt@3e18
    "t1": (16, 896, 7),     # M~1.63e9  near M_opt@1e19
    "t2": (22, 1024, 8),    # M~2.77e9  near M_opt@3e19
    "t3": (12, 1792, 14),   # M~3.83e9  bridge
    "t4": (24, 1536, 12),   # M~5.89e9  upper bridge (still left of s1)
    # Wave L4: left of t0 (head_dim 128). D=C/M must fit the frozen ~40B unique set.
    "u0": (6, 384, 3),      # M~1.77e8  C=3e18 only
    "u1": (8, 512, 4),      # M~3.52e8  C=3e18,1e19
    "u2": (10, 640, 5),     # M~6.09e8  C=3e18,1e19
    "u3": (10, 768, 6),     # M~8.02e8  all three C (only width left of t0 that fits C=3e19)
}

WAVE_A_ARCHS = ("s1", "s2", "s3", "s4", "s5", "s6")
WAVE_L_ARCHS = ("t0", "t1", "t2", "t3", "t4")
WAVE_L4_ARCHS = ("u0", "u1", "u2", "u3")

N_GPUS: dict[str, int] = {k: 8 for k in ARCHS}


def make_arch(arch_id: str, seq_len: int = 4096, n_vocab: int = 50257) -> Arch:
    n_layer, d_model, n_heads = ARCHS[arch_id]
    assert d_model % n_heads == 0
    return Arch(arch_id, n_layer, d_model, n_heads, seq_len, n_vocab)
