"""SwiGLU expert FFN + GShard / DeepSeekMoE layers (arXiv:2401.06066)."""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.distributed.nn import functional as dist_nn
except ImportError:  # pragma: no cover
    dist_nn = None


def _sync_lb_stats(
    hits: torch.Tensor,
    scores: torch.Tensor,
    n_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Global hits / routing scores for load-balance loss (DDP all-reduce)."""
    if not (dist.is_available() and dist.is_initialized()):
        s = float(max(n_tokens, 1))
        return hits, scores.sum(dim=0), s

    device = hits.device
    with torch.no_grad():
        hits_g = hits.to(torch.float32).clone()
        dist.all_reduce(hits_g, op=dist.ReduceOp.SUM)

    score_sum = scores.sum(dim=0).to(torch.float32)
    s_t = torch.tensor([float(n_tokens)], device=device, dtype=torch.float32)
    if dist_nn is not None:
        score_sum = dist_nn.all_reduce(score_sum, op=dist.ReduceOp.SUM)
        s_t = dist_nn.all_reduce(s_t, op=dist.ReduceOp.SUM)
    else:  # pragma: no cover
        dist.all_reduce(score_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(s_t, op=dist.ReduceOp.SUM)

    s_global = max(float(s_t.item()), 1.0)
    return hits_g, score_sum, s_global


def _dist_summary(values: list[float]) -> dict[str, float]:
    import math

    if not values:
        return {"entropy": 0.0, "imbalance": 0.0, "max": 0.0, "mean": 0.0}
    total = float(sum(values))
    if total <= 0.0:
        return {"entropy": 0.0, "imbalance": 0.0, "max": 0.0, "mean": 0.0}
    norm = [v / total for v in values if v > 0.0]
    entropy = float(-sum(v * math.log(v) for v in norm))
    mx = float(max(values))
    mean = total / len(values)
    return {
        "entropy": entropy,
        "imbalance": mx / max(mean, 1e-12),
        "max": mx,
        "mean": mean,
    }


def _topk_pairs(values: list[float], k: int = 8) -> list[list[float]]:
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)[:k]
    return [[float(i), float(values[i])] for i in order]


def ffn_hidden(d_model: int) -> int:
    """SwiGLU inner dim: 8/3 d, rounded to multiple of 64."""
    raw = (8.0 / 3.0) * d_model
    return max(64, int(round(raw / 64.0) * 64))


class ExpertFFN(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class BatchedExperts(nn.Module):
    """E independent SwiGLU experts as 3 stacked GEMMs: [S,d] x [E,h,d] -> [S,E,d]."""

    def __init__(self, n_experts: int, d_model: int, hidden: int):
        super().__init__()
        self.n_experts = n_experts
        self.d_model = d_model
        self.hidden = hidden
        self.w1 = nn.Parameter(torch.empty(n_experts, hidden, d_model))
        self.w2 = nn.Parameter(torch.empty(n_experts, d_model, hidden))
        self.w3 = nn.Parameter(torch.empty(n_experts, hidden, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [S, d] -> [S, E, d]
        a = torch.einsum("sd,ehd->seh", x, self.w1)
        b = torch.einsum("sd,ehd->seh", x, self.w3)
        h = F.silu(a) * b
        return torch.einsum("seh,edh->sed", h, self.w2)


class MoELayer(nn.Module):
    """Top-k routed experts, optional always-on shared expert. No token drop."""

    def __init__(
        self,
        d_model: int,
        n_routed: int,
        n_shared: int,
        k_routed: int,
        hidden: int,
        balance_alpha: float = 0.01,
    ):
        super().__init__()
        if k_routed < 1 or k_routed > n_routed:
            raise ValueError(f"k_routed={k_routed} not in 1..{n_routed}")
        self.n_routed = n_routed
        self.n_shared = n_shared
        self.k_routed = k_routed
        self.hidden = hidden
        self.balance_alpha = balance_alpha
        self.centroids = nn.Parameter(torch.empty(n_routed, d_model))
        self.routed = BatchedExperts(n_routed, d_model, hidden)
        self.shared = nn.ModuleList(ExpertFFN(d_model, hidden) for _ in range(n_shared))
        self.last_max_frac = 0.0
        self.last_routing: dict[str, list[float]] = {}
        # Eval-time overrides (None = use constructor values). Weights stay loaded.
        self.override_k_routed: int | None = None
        self.override_n_shared: int | None = None

    def expert_params(self) -> int:
        one = 3 * self.routed.d_model * self.routed.hidden
        return one * (self.n_routed + self.n_shared)

    def active_k_routed(self) -> int:
        k = self.k_routed if self.override_k_routed is None else int(self.override_k_routed)
        if k < 1 or k > self.n_routed:
            raise ValueError(f"active k_routed={k} not in 1..{self.n_routed}")
        return k

    def active_n_shared(self) -> int:
        n = self.n_shared if self.override_n_shared is None else int(self.override_n_shared)
        if n < 0 or n > self.n_shared:
            raise ValueError(f"active n_shared={n} not in 0..{self.n_shared}")
        return n

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        s = b * t
        flat = x.reshape(s, d)
        compute_dtype = flat.dtype
        k_routed = self.active_k_routed()
        n_shared = self.active_n_shared()
        shared_out = torch.zeros(s, d, device=flat.device, dtype=compute_dtype)
        for exp in self.shared[:n_shared]:
            shared_out = shared_out + exp(flat).to(dtype=compute_dtype)

        logits = flat.float().matmul(self.centroids.float().t())
        scores = torch.softmax(logits, dim=-1)
        topk_w, topk_idx = torch.topk(scores, k_routed, dim=-1)
        gates = torch.zeros(s, self.n_routed, device=flat.device, dtype=compute_dtype)
        gates.scatter_(1, topk_idx, topk_w.to(dtype=compute_dtype))
        y = self.routed(flat).to(dtype=compute_dtype)
        routed_out = (y * gates.unsqueeze(-1)).sum(dim=1)
        hits = torch.zeros(self.n_routed, device=flat.device, dtype=torch.float32)
        hits.scatter_add_(
            0,
            topk_idx.reshape(-1),
            torch.ones(s * k_routed, device=flat.device, dtype=torch.float32),
        )

        hits_g, score_sum, s_global = _sync_lb_stats(hits, scores, s)
        f = (self.n_routed / float(k_routed)) * (hits_g.detach() / s_global)
        p = score_sum / s_global
        bal = (f * p).sum()
        with torch.no_grad():
            token_frac = (hits_g / (s_global * float(k_routed))).detach().cpu().tolist()
            gate_mean = p.detach().cpu().tolist()
            self.last_max_frac = float((hits_g.max() / s_global).detach().item())
            self.last_routing = {
                "token_frac": token_frac,
                "gate_mean": gate_mean,
                "token_stats": _dist_summary(token_frac),
                "gate_stats": _dist_summary(gate_mean),
            }
        return (shared_out + routed_out).view(b, t, d), bal
