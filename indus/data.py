"""Data loading utilities: memmap-backed token batches for training."""

import json
import os

import numpy as np
import torch


class TokenDataset:
    """Reads uint16 token binaries produced by scripts/prepare_data.py."""

    def __init__(self, bin_path: str):
        meta_path = bin_path + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.n_tokens = int(json.load(f)["n_tokens"])
        else:
            self.n_tokens = None
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        assert len(self.data) >= 2, f"{bin_path} has too few tokens"

    @property
    def size(self) -> int:
        return len(self.data)


def get_batch(dataset: TokenDataset, block_size: int, batch_size: int,
              device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random (inputs, targets) windows; targets = inputs shifted by 1."""
    ix = torch.randint(0, dataset.size - block_size - 1, (batch_size,))
    x = torch.stack([
        torch.from_numpy(dataset.data[i:i + block_size].astype(np.int64))
        for i in ix])
    y = torch.stack([
        torch.from_numpy(dataset.data[i + 1:i + 1 + block_size].astype(np.int64))
        for i in ix])
    return x.to(device), y.to(device)


class SFTDataset:
    """Token ids + aligned loss mask produced by scripts/build_sft_data.py."""

    def __init__(self, bin_path: str):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        mask_path = bin_path.replace(".bin", ".mask.bin")
        if os.path.exists(mask_path):
            self.mask = np.memmap(mask_path, dtype=np.uint8, mode="r")
        else:
            self.mask = np.ones(len(self.data), dtype=np.uint8)
        assert len(self.mask) == len(self.data)

    @property
    def size(self) -> int:
        return len(self.data)


def get_sft_batch(dataset: SFTDataset, block_size: int, batch_size: int,
                  device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Like get_batch but returns targets with non-assistant positions masked
    to -100 so the loss only trains on assistant responses."""
    ix = torch.randint(0, max(1, dataset.size - block_size - 1), (batch_size,))
    xs, ys = [], []
    for i in ix:
        x = torch.from_numpy(dataset.data[i:i + block_size].astype(np.int64))
        m = torch.from_numpy(dataset.mask[i + 1:i + 1 + block_size]
                             .astype(np.int64))
        y = torch.from_numpy(dataset.data[i + 1:i + 1 + block_size]
                             .astype(np.int64)).masked_fill(m == 0, -100)
        xs.append(x)
        ys.append(y)
    return (torch.stack(xs).to(device), torch.stack(ys).to(device))
