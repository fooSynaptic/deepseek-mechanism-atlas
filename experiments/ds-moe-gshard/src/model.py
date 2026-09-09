"""V1 block with MoE FFN: RMSNorm, SwiGLU experts, RoPE, MHA."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from moe import BatchedExperts, MoELayer, _topk_pairs, ffn_hidden  # noqa: F401 — re-exported for train.py


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


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, moe: MoELayer):
        super().__init__()
        self.n1 = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.n2 = RMSNorm(d_model)
        self.ffn = moe

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = x + self.attn(self.n1(x), cos, sin)
        y, bal = self.ffn(self.n2(x))
        return x + y, bal


def round_hidden(h: float) -> int:
    """Round intermediate dim up/down to multiple of 64 (same rule as ffn_hidden)."""
    return max(64, int(round(h / 64.0) * 64))


def make_moe(arch_id: str, d_model: int, balance_alpha: float) -> MoELayer:
    h_full = ffn_hidden(d_model)
    if arch_id == "gshard":
        return MoELayer(
            d_model,
            n_routed=16,
            n_shared=0,
            k_routed=2,
            hidden=h_full,
            balance_alpha=balance_alpha,
        )
    if arch_id == "gshard_x15":
        # Paper Table 2: 1.5× expert width → ~1.5× expert params and activated FLOPs.
        return MoELayer(
            d_model,
            n_routed=16,
            n_shared=0,
            k_routed=2,
            hidden=round_hidden(h_full * 1.5),
            balance_alpha=balance_alpha,
        )
    if arch_id == "dsmoe":
        if h_full % 4 != 0:
            raise ValueError(f"ffn_hidden {h_full} not divisible by 4")
        return MoELayer(
            d_model,
            n_routed=63,
            n_shared=1,
            k_routed=7,
            hidden=h_full // 4,
            balance_alpha=balance_alpha,
        )
    if arch_id == "dsmoe_k3":
        # Paper Fig 6: same pool, train with Top-3 → half activated vs GShard.
        if h_full % 4 != 0:
            raise ValueError(f"ffn_hidden {h_full} not divisible by 4")
        return MoELayer(
            d_model,
            n_routed=63,
            n_shared=1,
            k_routed=3,
            hidden=h_full // 4,
            balance_alpha=balance_alpha,
        )
    if arch_id == "dsmoe_s0":
        # S-train: no shared expert from scratch. 64 quarter-experts, Top-8.
        # Same total + activated params as DeepSeekMoE 1+Top-7.
        if h_full % 4 != 0:
            raise ValueError(f"ffn_hidden {h_full} not divisible by 4")
        return MoELayer(
            d_model,
            n_routed=64,
            n_shared=0,
            k_routed=8,
            hidden=h_full // 4,
            balance_alpha=balance_alpha,
        )
    raise ValueError(f"unknown arch_id {arch_id}")


class MoELM(nn.Module):
    def __init__(
        self,
        arch_id: str,
        n_layer: int,
        d_model: int,
        n_heads: int,
        n_vocab: int,
        seq_len: int,
        init_std: float = 0.006,
        balance_alpha: float = 0.01,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.arch_id = arch_id
        self.n_layer = n_layer
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_vocab = n_vocab
        self.seq_len = seq_len
        self.head_dim = d_model // n_heads
        self.balance_alpha = balance_alpha
        self.gradient_checkpointing = False
        self.last_stats = {}

        self.tok = nn.Embedding(n_vocab, d_model)
        self.blocks = nn.ModuleList(
            Block(d_model, n_heads, make_moe(arch_id, d_model, balance_alpha))
            for _ in range(n_layer)
        )
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, n_vocab, bias=False)
        self.lm_head.weight = self.tok.weight
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.apply(lambda m: _init(m, init_std))

    def expert_params(self) -> int:
        return sum(blk.ffn.expert_params() for blk in self.blocks)

    def dense_ffn_params(self) -> int:
        h = ffn_hidden(self.d_model)
        one = 3 * self.d_model * h
        return one * self.n_layer

    def set_routing_override(
        self,
        k_routed: int | None = None,
        n_shared: int | None = None,
    ) -> None:
        """Eval-time routing surgery. Pass None to clear that override."""
        for blk in self.blocks:
            blk.ffn.override_k_routed = k_routed
            blk.ffn.override_n_shared = n_shared

    def _rope(self, t: int, device, dtype):
        pos = torch.arange(t, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        b, t = idx.shape
        x = self.tok(idx)
        cos, sin = self._rope(t, x.device, x.dtype)
        bals = []
        max_frac = 0.0
        layer_max_frac: list[float] = []
        for blk in self.blocks:
            if self.gradient_checkpointing and self.training:
                x, bal = torch.utils.checkpoint.checkpoint(
                    blk, x, cos, sin, use_reentrant=False
                )
            else:
                x, bal = blk(x, cos, sin)
            bals.append(bal)
            layer_max_frac.append(float(blk.ffn.last_max_frac))
            max_frac = max(max_frac, blk.ffn.last_max_frac)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        ce = None
        aux = None
        if targets is not None:
            ce = F.cross_entropy(logits.view(-1, self.n_vocab), targets.view(-1))
            aux = self.balance_alpha * torch.stack(bals).sum()
            loss = ce + aux
            self.last_stats.clear()
            stats: dict[str, object] = {
                "ce": float(ce.detach()),
                "aux": float(aux.detach()),
                "max_expert_frac": max_frac,
                "aux_ce_ratio": float(aux.detach()) / max(float(ce.detach()), 1e-9),
                "layer_max_frac": layer_max_frac,
            }
            mid = self.n_layer // 2
            routing = getattr(self.blocks[mid].ffn, "last_routing", None) or {}
            token_frac = routing.get("token_frac") or []
            gate_mean = routing.get("gate_mean") or []
            token_stats = routing.get("token_stats") or {}
            gate_stats = routing.get("gate_stats") or {}
            if token_frac and gate_mean:
                stats.update(
                    {
                        "routing_layer": mid,
                        "load_imbalance": float(token_stats.get("imbalance", 0.0)),
                        "token_frac_entropy": float(token_stats.get("entropy", 0.0)),
                        "gate_entropy": float(gate_stats.get("entropy", 0.0)),
                        "token_frac_top": _topk_pairs(token_frac, 8),
                        "gate_mean_top": _topk_pairs(gate_mean, 8),
                    }
                )
            self.last_stats.update(stats)
        return logits, loss


def _init(module: nn.Module, std: float) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif isinstance(module, MoELayer):
        nn.init.normal_(module.centroids, mean=0.0, std=std)
    elif isinstance(module, BatchedExperts):
        nn.init.normal_(module.w1, mean=0.0, std=std)
        nn.init.normal_(module.w2, mean=0.0, std=std)
        nn.init.normal_(module.w3, mean=0.0, std=std)
