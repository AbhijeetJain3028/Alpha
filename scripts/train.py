#!/usr/bin/env python3
"""Train the Indus language model.

Examples:
  # tiny preset on CPU with defaults:
  python scripts/train.py --data-dir data --preset indus-nano --max-iters 2000

  # larger preset on GPU with overrides:
  python scripts/train.py --data-dir data --preset indus-small \
      --batch-size 32 --max-iters 20000 --device cuda

Checkpoints are written to --out-dir as ckpt.pt (weights + config).
"""

import argparse
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indus.config import get_config, PRESETS           # noqa: E402
from indus.data import TokenDataset, get_batch          # noqa: E402
from indus.model import IndusLM                         # noqa: E402


@torch.no_grad()
def estimate_loss(model, datasets: dict, cfg, eval_iters: int,
                  batch_size: int, device: str) -> dict:
    model.eval()
    out = {}
    for split, ds in datasets.items():
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = get_batch(ds, cfg.block_size, batch_size, device)
            out_ = model(x, targets=y)
            losses[k] = out_.loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def lr_at(step: int, max_iters: int, warmup: int, base_lr: float,
          min_lr_ratio: float) -> float:
    """Linear warmup followed by cosine decay."""
    if step < warmup:
        return base_lr * (step + 1) / warmup
    if step >= max_iters:
        return base_lr * min_lr_ratio
    ratio = (step - warmup) / (max_iters - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return base_lr * min_lr_ratio + coeff * base_lr * (1 - min_lr_ratio)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    g = ap.add_argument_group("model overrides")
    g.add_argument("--n-layer", type=int, default=None)
    g.add_argument("--n-head", type=int, default=None)
    g.add_argument("--n-kv-head", type=int, default=None)
    g.add_argument("--n-embd", type=int, default=None)
    g.add_argument("--block-size", type=int, default=None)
    g.add_argument("--vocab-size", type=int, default=None)

    t = ap.add_argument_group("training")
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--max-iters", type=int, default=5000)
    t.add_argument("--lr", type=float, default=None)
    t.add_argument("--warmup", type=int, default=100)
    t.add_argument("--weight-decay", type=float, default=0.1)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--dropout", type=float, default=0.0)
    t.add_argument("--eval-interval", type=int, default=250)
    t.add_argument("--eval-iters", type=int, default=20)
    t.add_argument("--log-interval", type=int, default=10)
    t.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    tok_path = os.path.join(args.data_dir, "tokenizer.json")
    vocab_size = args.vocab_size
    if vocab_size is None and os.path.exists(tok_path):
        from indus.tokenizer import BPETokenizer
        vocab_size = len(BPETokenizer.load(tok_path).vocab)

    overrides = dict(
        n_layer=args.n_layer, n_head=args.n_head, n_kv_head=args.n_kv_head,
        n_embd=args.n_embd, block_size=args.block_size, dropout=args.dropout,
        weight_decay=args.weight_decay, vocab_size=vocab_size,
    )
    cfg = get_config(args.preset, **overrides)
    assert cfg.vocab_size is not None, "vocab size unknown; run prepare_data first"
    device = args.device

    print(f"config: {cfg.to_dict()}")
    model = IndusLM(cfg).to(device)
    print(f"parameters: {model.num_params() / 1e6:.2f}M | device: {device}")

    train_ds = TokenDataset(os.path.join(args.data_dir, "train.bin"))
    val_path = os.path.join(args.data_dir, "val.bin")
    datasets = {"train": train_ds}
    if os.path.exists(val_path):
        datasets["val"] = TokenDataset(val_path)
    print(f"train tokens: {train_ds.size:,}")

    lr = args.lr if args.lr is not None else (3e-3 if cfg.n_embd <= 128 else 6e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                  betas=(0.9, 0.95), eps=1e-8,
                                  weight_decay=cfg.weight_decay)

    os.makedirs(args.out_dir, exist_ok=True)
    model.train()
    t0 = time.time()
    best_val = float("inf")

    use_amp = device.startswith("cuda")
    # bf16 on Ampere+, fp16 (+GradScaler) on older GPUs like P100/T4
    amp_dtype = torch.bfloat16
    if use_amp and torch.cuda.get_device_capability(0)[0] < 8:
        amp_dtype = torch.float16
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    for step in range(args.max_iters):
        lr_step = lr_at(step, args.max_iters, args.warmup, lr, 0.1)
        for group in optimizer.param_groups:
            group["lr"] = lr_step

        x, y = get_batch(train_ds, cfg.block_size, args.batch_size, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype,
                            enabled=use_amp):
            out = model(x, targets=y)
        loss = out.loss

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % args.log_interval == 0 or step == args.max_iters - 1:
            dt = time.time() - t0
            tps = args.batch_size * cfg.block_size * (step + 1) / max(dt, 1e-6)
            print(f"iter {step:5d}/{args.max_iters} | loss {loss.item():.4f} "
                  f"| ppl {math.exp(loss.item()):7.2f} | lr {lr_step:.2e} "
                  f"| {tps / 1e3:.1f}k tok/s | {dt:.0f}s")

        if (step > 0 and step % args.eval_interval == 0) \
                or step == args.max_iters - 1:
            losses = estimate_loss(model, datasets, cfg, args.eval_iters,
                                   args.batch_size, device)
            msg = " | ".join(f"{s} loss {l:.4f}" for s, l in losses.items())
            print(f"[eval ] iter {step}: {msg}")
            if losses.get("val", losses["train"]) < best_val:
                best_val = losses.get("val", losses["train"])
                ckpt = {
                    "model": model.state_dict(),
                    "config": cfg.to_dict(),
                    "step": step,
                    "val_loss": best_val,
                }
                path = os.path.join(args.out_dir, "ckpt.pt")
                torch.save(ckpt, path)
                print(f"[save ] best checkpoint -> {path}")

    print("\ntraining complete.")


if __name__ == "__main__":
    main()
