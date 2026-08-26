"""Muon optimizer - orthogonalized momentum for 2D+ parameters.

Ported per docs/PLAYBOOK.md §4 from Jordan (2024)'s Muon, with the
standard hybrid split used by its production adopters:

    ndim >= 2 params  -> Muon: momentum + Newton-Schulz-5 orthogonalized
                         update (steepest descent under the spectral norm)
    ndim < 2 params   -> AdamW fallback (embeddings, norms, gates)

The Newton-Schulz iteration approximates the polar factor U V^T of the
update matrix in ~5 bfloat16 matmuls-per-step, giving each matrix update
approximately equal singular values - empirically ~1.35-2x faster
optimization than AdamW on transformer matrices at small scale, and
validated at multi-trillion-token scale (Kimi K2, Moonshot '25).

Zero dependencies beyond PyTorch.
"""

from __future__ import annotations

import torch

from .config import IndusConfig


# ------------------------------------------------------------- newton-schulz
@torch.no_grad()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Approximate G @ (G^T G)^{-1/2}: all singular values -> ~1."""
    assert G.ndim == 2, "newton-schulz needs matrices"
    a, b, c = 3.4445, -4.7750, 2.0315          # quintic coefficients
    X = G.bfloat16()
    transposed = G.size(0) > G.size(1)         # iterate on the skinny side
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """Pure-Muon optimizer (use HybridMuonAdamW for real models)."""

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 nesterov: bool = True, ns_steps: int = 5,
                 weight_decay: float = 0.0,
                 rms_matched: bool = True):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay,
                        rms_matched=rms_matched)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):                      # noqa: D102
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            nes, nsw = group["nesterov"], group["ns_steps"]
            wd, rms_m = group["weight_decay"], group["rms_matched"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"]
                buf.mul_(mom).add_(p.grad)
                g = p.grad.lerp_(buf, mom) if nes else buf
                u = zeropower_via_newtonschulz5(g, steps=nsw).to(p.dtype)
                # scale so update RMS is independent of matrix shape
                if rms_m:
                    u *= max(1.0, p.size(-2) / p.size(-1)) ** 0.5
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(u, alpha=-lr)
        return loss


class HybridMuonAdamW:
    """Facade: Muon on >=2D weights, AdamW on everything else.

    Usage mirrors torch.optim:
        opt = HybridMuonAdamW(model.parameters(), muon_lr=0.02,
                              adamw_lr=6e-4, weight_decay=0.1)
        opt.zero_grad(); loss.backward(); opt.step()
    """

    def __init__(self, params, muon_lr: float = 0.02, adamw_lr: float = 3e-4,
                  betas=(0.9, 0.95), eps: float = 1e-8,
                  weight_decay: float = 0.1, momentum: float = 0.95,
                  ns_steps: int = 5, rms_matched: bool = True):
        matrix, vector = [], []
        for p in params:
            (matrix if p.ndim >= 2 else vector).append(p)
        self.matrix_params = matrix
        self.vector_params = vector
        self.muon = Muon(matrix, lr=muon_lr, momentum=momentum,
                         ns_steps=ns_steps, rms_matched=rms_matched) \
            if matrix else None
        # decoupled WD only where it applies; norms/1D typically wd-free but
        # keep parity with our AdamW recipe unless told otherwise
        self.adamw = torch.optim.AdamW(vector, lr=adamw_lr, betas=betas,
                                       eps=eps, weight_decay=weight_decay) \
            if vector else None

    # pass-through API -----------------------------------------------------
    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in (self.muon, self.adamw):
            if opt:
                opt.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for opt in (self.muon, self.adamw):
            if opt:
                opt.step()

    def param_groups(self):
        groups = []
        for opt in (self.muon, self.adamw):
            if opt:
                groups.extend(opt.param_groups)
        return groups

    def state_dict(self) -> dict:
        return {"muon": self.muon.state_dict() if self.muon else None,
                "adamw": self.adamw.state_dict() if self.adamw else None}

    def load_state_dict(self, sd: dict) -> None:
        if self.muon and sd.get("muon"):
            self.muon.load_state_dict(sd["muon"])
        if self.adamw and sd.get("adamw"):
            self.adamw.load_state_dict(sd["adamw"])


def build_optimizer(cfg: IndusConfig, model: torch.nn.Module,
                    muon_lr: float = 0.02, adamw_lr: float = 3e-4,
                    **kw) -> HybridMuonAdamW:
    """Recipe hook for training scripts: use_muon flag switches stack."""
    if getattr(cfg, "use_muon", False):
        return HybridMuonAdamW(model.parameters(), muon_lr=muon_lr,
                               adamw_lr=adamw_lr, **kw)
    return torch.optim.AdamW(model.parameters(), lr=adamw_lr,
                             betas=(0.9, 0.95), eps=1e-8,
                             weight_decay=cfg.weight_decay)   # type: ignore[return-value]
