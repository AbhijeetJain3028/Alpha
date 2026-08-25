#!/usr/bin/env python3
"""Supervised fine-tuning (chat) stage for Indus.

Loads the pretrained base checkpoint, registers chat special tokens
(<|system|> <|user|> <|assistant|> <|end|>) by resizing the embeddings, and
trains on assistant responses only (loss-masked prompts).

Designed to run either locally or inside a Kaggle kernel:
  python scripts/train_sft.py --base-ckpt ckpt-latest.pt --data-dir data_sft

The base checkpoint can be pulled straight from the Hub with --from-hub.
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indus.config import IndusConfig                 # noqa: E402
from indus.data import SFTDataset, get_sft_batch     # noqa: E402
from indus.model import IndusLM                      # noqa: E402
from indus.tokenizer import BPETokenizer             # noqa: E402


def load_base(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = IndusConfig.from_dict(ckpt["config"])
    return cfg, ckpt


def resize_for_vocab(model: IndusLM, old_state: dict, new_vocab: int) -> None:
    """Copy matching weights; keep fresh random init for new token rows."""
    state = model.state_dict()
    loaded = 0
    for k, v in old_state.items():
        if k not in state or state[k].shape != v.shape:
            continue
        state[k] = v
        loaded += 1
    model.load_state_dict(state)
    print(f"[init] loaded {loaded}/{len(state)} tensors "
          f"(embedding resized {old_state['tok_emb.weight'].shape[0]} -> {new_vocab})")


@torch.no_grad()
def quick_chat(model, tok, device, prompt="What is the capital of France?"):
    ids = tok.encode_with_specials(f"<|user|>\n{prompt}<|end|>\n<|assistant|>\n")
    x = torch.tensor([ids], device=device)
    y = model.generate(x, max_new_tokens=80, temperature=0.7, top_k=50,
                       endoftext_id=tok.special_tokens.get("<|endoftext|>"))
    text = tok.decode(y[0].tolist())
    # specials decode as stripped; show raw-ish reply between markers
    print("[sample]", repr(text[-160:]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-ckpt", default=None)
    ap.add_argument("--tokenizer", default="data_v2/tokenizer.json")
    ap.add_argument("--data-dir", default="data_sft")
    ap.add_argument("--out", default="checkpoints/ckpt-sft.pt")
    ap.add_argument("--hf-repo", default=os.environ.get("HF_REPO",
                   "AbhijeetJain4075/indus-llm"))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.base_ckpt is None and args.hf_repo:
        from huggingface_hub import hf_hub_download
        args.base_ckpt = hf_hub_download(args.hf_repo, "ckpt-latest.pt",
                                         repo_type="model")
        print(f"[hub ] base ckpt: {args.base_ckpt}")

    tok = BPETokenizer.load(args.tokenizer)
    tok.add_chat_specials()
    tok.save(args.tokenizer)

    cfg, base = load_base(args.base_ckpt)
    cfg.vocab_size = len(tok.vocab)               # grow for chat specials
    device = args.device

    model = IndusLM(cfg).to(device)
    resize_for_vocab(model, {k: v.cpu() for k, v in base["model"].items()},
                     cfg.vocab_size)
    del base

    train_ds = SFTDataset(os.path.join(args.data_dir, "sft_train.bin"))
    val_ds = SFTDataset(os.path.join(args.data_dir, "sft_val.bin"))
    print(f"[data] sft train tokens: {train_ds.size:,} | "
          f"val: {val_ds.size:,} | vocab: {cfg.vocab_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=0.1)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        r = min(1.0, step / args.steps)
        return args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * r)))

    use_amp = device.startswith("cuda")
    amp_dtype = torch.bfloat16 if (use_amp and
                                   torch.cuda.get_device_capability(0)[0] >= 8) \
        else torch.float16
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    @torch.no_grad()
    def eval_loss(iters=20):
        model.eval()
        losses = []
        for _ in range(iters):
            x, y = get_sft_batch(val_ds, cfg.block_size, args.batch_size, device)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                losses.append(model(x, targets=y).loss.item())
        model.train()
        return sum(losses) / len(losses)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model.train()
    t0 = time.time()
    for step in range(args.steps):
        lr = lr_at(step)
        for g in optimizer.param_groups:
            g["lr"] = lr
        x, y = get_sft_batch(train_ds, cfg.block_size, args.batch_size, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
            out = model(x, targets=y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(out.loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step % 25 == 0 or step == args.steps - 1:
            el = time.time() - t0
            print(f"sft iter {step:5d}/{args.steps} | loss {out.loss.item():.4f} | "
                  f"lr {lr:.2e} | {el / max(step + 1, 1):.2f}s/it", flush=True)
        if (step + 1) % args.eval_every == 0:
            print(f"[eval] sft val loss {eval_loss():.4f}")
            quick_chat(model, tok, device)
        if ((step + 1) % args.save_every == 0 or step == args.steps - 1):
            payload = {"model": model.state_dict(), "config": cfg.to_dict(),
                       "step": step + 1, "kind": "sft"}
            tmp = args.out + ".tmp"
            torch.save(payload, tmp)
            os.replace(tmp, args.out)
            print(f"[save] {args.out}")
            if args.hf_repo and os.environ.get("HF_TOKEN"):
                try:
                    from huggingface_hub import HfApi
                    HfApi(token=os.environ["HF_TOKEN"]).upload_file(
                        path_or_fileobj=args.out, path_in_repo="ckpt-sft.pt",
                        repo_id=args.hf_repo, repo_type="model",
                        commit_message=f"SFT step {step + 1}")
                    print("[hub ] uploaded ckpt-sft.pt")
                except Exception as e:
                    print("[hub ] upload failed:", e)

    print("\n[done] chat with: python scripts/generate.py "
          "--ckpt checkpoints/ckpt-sft.pt --chat")


if __name__ == "__main__":
    main()
