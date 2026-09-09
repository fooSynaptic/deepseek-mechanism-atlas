"""V1-like dense decoder: RMSNorm, SwiGLU, RoPE, MHA (no GQA)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flops import ffn_hidden


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(orig)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q,k: (B, H, T, D) ; cos/sin: (T, D)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = (q * cos) + (rotate_half(q) * sin)
    k = (k * cos) + (rotate_half(k) * sin)
    return q, k


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.wqkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rope(q, k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(b, t, d)
        return self.wo(y)


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        h = ffn_hidden(d_model)
        self.w1 = nn.Linear(d_model, h, bias=False)
        self.w2 = nn.Linear(h, d_model, bias=False)
        self.w3 = nn.Linear(d_model, h, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n1 = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.n2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x), cos, sin)
        x = x + self.ffn(self.n2(x))
        return x


class V1DenseLM(nn.Module):
    def __init__(
        self,
        n_layer: int,
        d_model: int,
        n_heads: int,
        n_vocab: int,
        seq_len: int,
        init_std: float = 0.006,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_layer = n_layer
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_vocab = n_vocab
        self.seq_len = seq_len
        self.head_dim = d_model // n_heads

        self.tok = nn.Embedding(n_vocab, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_heads) for _ in range(n_layer))
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, n_vocab, bias=False)
        self.lm_head.weight = self.tok.weight
        self.gradient_checkpointing = False

        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.apply(lambda m: _init(m, init_std))

    def _rope(self, t: int, device, dtype):
        pos = torch.arange(t, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        x = self.tok(idx)
        cos, sin = self._rope(t, x.device, x.dtype)
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    blk, x, cos, sin, use_reentrant=False
                )
            else:
                x = blk(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.n_vocab), targets.view(-1))
        return logits, loss


def _init(module: nn.Module, std: float) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
