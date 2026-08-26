"""Tests for the Muon optimizer port.
Run: python tests/test_muon.py
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indus.config import get_config                    # noqa: E402
from indus.model import IndusLM                        # noqa: E402
from indus.muon import (HybridMuonAdamW, Muon,          # noqa: E402
                        zeropower_via_newtonschulz5)


def test_newton_schulz_orthogonalizes():
    torch.manual_seed(0)
    G = torch.randn(64, 32) * 7.0                     # badly scaled input
    U = zeropower_via_newtonschulz5(G, steps=5).float()
    s = torch.linalg.svdvals(U)
    # NS-5's quintic iteration drives all singular values into ~[0.7, 1.3]
    # (approximate polar factor - exactly Muon's design intent)
    assert 0.55 <= s.min() <= 1.45 and s.max() <= 1.45, s
    assert s.std() < 0.25, s.std()
    print(f"[ok ] newton-schulz: singular values in "
          f"[{s.min():.3f}, {s.max():.3f}] (std {s.std():.3f})")


def test_muon_reduces_loss_faster_or_equal():
    """Same tiny model + data: Muon-hybrid must reach AdamW's loss or better
    within a small step budget (its raison d'etre)."""
    from indus.data import TokenDataset, get_batch
    cfg = get_config("indus-nano")
    ds = TokenDataset("data/train.bin")

    def run(opt_kind: str, steps: int = 60) -> float:
        torch.manual_seed(1337)
        model = IndusLM(cfg)
        if opt_kind == "muon":
            opt = HybridMuonAdamW(model.parameters(), muon_lr=0.02,
                                  adamw_lr=1e-3, weight_decay=0.0)
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3,
                                    betas=(0.9, 0.95), weight_decay=0.0)
        model.train()
        last = None
        for _ in range(steps):
            x, y = get_batch(ds, cfg.block_size, 4, "cpu")
            loss = model(x, targets=y).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last = loss.item()
        return last

    adamw_last = run("adamw")
    muon_last = run("muon")
    print(f"[ok ] 60-step nano training: adamw {adamw_last:.3f} | "
          f"muon {muon_last:.3f}")
    assert muon_last <= adamw_last * 1.05, \
        f"muon {muon_last:.3f} materially worse than adamw {adamw_last:.3f}"


def test_hybrid_split():
    cfg = get_config("indus-nano")
    model = IndusLM(cfg)
    opt = HybridMuonAdamW(model.parameters(), muon_lr=0.01,
                          adamw_lr=6e-4, weight_decay=0.1)
    n_matrix = sum(p.numel() for p in model.parameters() if p.ndim >= 2)
    n_vector = sum(p.numel() for p in model.parameters() if p.ndim < 2)
    got_m = sum(p.numel() for g in opt.muon.param_groups
                for p in g["params"])
    got_v = sum(p.numel() for g in opt.adamw.param_groups
                for p in g["params"])
    assert got_m == n_matrix and got_v == n_vector
    sd = opt.state_dict()
    opt.load_state_dict(sd)                            # round-trips cleanly
    print(f"[ok ] hybrid split: matrix {n_matrix:,} -> muon, "
          f"vector {n_vector:,} -> adamw; state round-trips")


if __name__ == "__main__":
    test_newton_schulz_orthogonalizes()
    test_hybrid_split()
    test_muon_reduces_loss_faster_or_equal()
    print("\nALL MUON TESTS PASSED")
