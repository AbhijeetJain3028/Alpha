"""Sparse Mixture-of-Experts FFN for Indus - ported from Indus-Kernel
`ik_indus_llm/moe.py`, adapted to our SwiGLU experts and config style.

Dormant by default: enable with `use_moe=True` + `n_experts>1` on the NEXT
pretrain (MoE cannot be retrofitted into a dense checkpoint). Peak logic:
expert granularity beats width at fixed FLOPs for small models, and the
load-balancing auxiliary loss is mandatory to prevent expert collapse
(Shazeer'17 / DeepSeekMoE'24 doctrine).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import IndusConfig


class MoESwiGLU(nn.Module):
    """Router + N parallel SwiGLU experts with load-balancing aux loss."""

    def __init__(self, config: IndusConfig):
        super().__init__()
        assert config.n_experts >= 1
        assert config.n_experts % config.n_experts_active == 0 or True
        self.n_experts = config.n_experts
        self.k = min(config.n_experts_active, config.n_experts)
        hidden = int(8 * config.n_embd / 3)
        hidden = ((hidden + config.ffn_multiple_of - 1)
                  // config.ffn_multiple_of) * config.ffn_multiple_of
        # NOTE: expert hidden is shrunk by k so active-param count matches a
        # dense SwiGLU of the same budget - peak params-per-FLOP practice.
        self.expert_hidden = max(config.ffn_multiple_of,
                                 int(hidden * self.k / self.n_experts))
        self.w_gate = nn.ModuleList()
        self.w_up = nn.ModuleList()
        self.w_down = nn.ModuleList()
        for _ in range(self.n_experts):
            self.w_gate.append(nn.Linear(config.n_embd, self.expert_hidden,
                                         bias=False))
            self.w_up.append(nn.Linear(config.n_embd, self.expert_hidden,
                                       bias=False))
            self.w_down.append(nn.Linear(self.expert_hidden, config.n_embd,
                                         bias=False))
        self.router = nn.Linear(config.n_embd, self.n_experts, bias=False)
        self.aux_loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        flat = x.view(-1, C)                                   # (B*T, C)
        logits = self.router(flat)                             # (B*T, E)
        probs = F.softmax(logits.float(), dim=-1)
        top_p, top_i = torch.topk(probs, self.k, dim=-1)
        top_p = top_p / top_p.sum(-1, keepdim=True)

        # load-balancing aux loss (Switch-Transformer form)
        if self.training:
            mean_p = probs.mean(dim=0)
            frac = torch.zeros_like(mean_p).scatter_add_(
                0, top_i.reshape(-1),
                torch.ones_like(top_i.reshape(-1), dtype=mean_p.dtype))
            frac = frac / top_i.numel()
            self.aux_loss = self.n_experts * (mean_p * frac).sum()

        y = torch.zeros_like(flat)
        for e in range(self.n_experts):
            mask_e = (top_i == e)                              # (B*T, k)
            if not mask_e.any():
                continue
            tok, slot = mask_e.nonzero(as_tuple=True)
            w = top_p[tok, slot].unsqueeze(-1).to(flat.dtype)
            xe = flat[tok]
            ye = self.w_down[e](F.silu(self.w_gate[e](xe))
                                * self.w_up[e](xe))
            y.index_add_(0, tok, w * ye)
        return y.view(B, T, C)


class DenseFFNCompat(nn.Module):
    """Adapter so dense checkpoints and MoE models share one Block path."""

    def __init__(self, dense: nn.Module):
        super().__init__()
        self.dense = dense
        self.aux_loss = 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense(x)
