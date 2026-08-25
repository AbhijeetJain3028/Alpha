#!/usr/bin/env python3
"""Online learning for Indus: fine-tune from user feedback.

Reads data_feedback/feedback.jsonl (written by the web UI 👍/👎/"teach a
better answer" flow and the CLI /good //bad //teach commands) and produces
ckpt-online.pt — the same base model updated toward:

  - replies users rated up          (reinforced as-is)
  - "taught" corrections            (user text replaces the reply)

This is human-in-the-loop continual learning: nothing synthetic, only what
real users actually said. Run it after any feedback session:

  python scripts/train_from_feedback.py --steps 80
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indus.config import IndusConfig                 # noqa: E402
from indus.model import IndusLM                      # noqa: E402
from indus.tokenizer import BPETokenizer             # noqa: E402

FEEDBACK_PATH = os.path.join("data_feedback", "feedback.jsonl")


def load_examples(tok):
    """Convert feedback records to (ids, mask) chat examples."""
    if not os.path.exists(FEEDBACK_PATH):
        return []
    sid = tok.special_tokens
    examples = []
    with open(FEEDBACK_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            verdict = r.get("verdict")
            if verdict == "down" and not r.get("correction"):
                continue                     # signal without target - skip
            target_text = r["correction"] if verdict == "taught" \
                else r.get("reply", "")
            if not r.get("prompt") or not target_text:
                continue

            ids, mask = [], []

            def emit(text_or_special, train_on, special=None):
                nonlocal ids, mask
                if special is not None:
                    ids.append(special)
                    mask.append(1 if train_on else 0)
                    return
                piece = tok.encode(text_or_special)
                ids.extend(piece)
                mask.extend([1 if train_on else 0] * len(piece))

            emit(None, False, special=sid["<|user|>"])
            emit("\n", False)
            emit(r["prompt"], False)
            emit(None, False, special=sid["<|end|>"])
            emit("\n", False)
            emit(None, False, special=sid["<|assistant|>"])
            emit("\n", False)
            emit(target_text, True)
            emit(None, True, special=sid["<|end|>"])
            emit("\n", False)
            ids.append(sid["<|endoftext|>"])
            mask.append(0)
            examples.append((ids, mask))
    return examples


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-ckpt", default=None,
                    help="defaults to newest of ckpt-sft/ckpt-latest local or Hub")
    ap.add_argument("--tokenizer", default="data_v2/tokenizer.json")
    ap.add_argument("--out", default="checkpoints/ckpt-online.pt")
    ap.add_argument("--hf-repo", default=os.environ.get("HF_REPO",
                   "AbhijeetJain4075/indus-llm"))
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    tok.add_chat_specials()

    examples = load_examples(tok)
    print(f"[fb  ] usable examples: {len(examples)}")
    if not examples:
        raise SystemExit("no feedback to learn from yet - rate some replies first")

    # resolve base checkpoint: newest local sft/online > hub latest
    base = args.base_ckpt
    if base is None:
        cands = ["checkpoints/ckpt-sft.pt", "checkpoints/ckpt-latest.pt"]
        cands = [c for c in cands if os.path.exists(c)]
        if cands:
            base = max(cands, key=os.path.getmtime)
        else:
            from huggingface_hub import hf_hub_download
            which = "ckpt-sft.pt" if args.hf_repo else "ckpt-latest.pt"
            try:
                base = hf_hub_download(args.hf_repo, "ckpt-sft.pt",
                                       repo_type="model")
            except Exception:
                base = hf_hub_download(args.hf_repo, "ckpt-latest.pt",
                                       repo_type="model")
    print(f"[base] {base}")
    ckpt = torch.load(base, map_location="cpu", weights_only=False)
    cfg = IndusConfig.from_dict(ckpt["config"])
    cfg.vocab_size = len(tok.vocab)
    device = args.device
    model = IndusLM(cfg).to(device)
    state = {k: v.cpu() for k, v in ckpt["model"].items()}
    final = {k: v for k, v in state.items()
             if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
    model.load_state_dict(final, strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=0.01)

    # pack examples into padded batches by similar length
    block = cfg.block_size

    def make_batch(exs):
        """Standard LM batching: inputs are ids[:-1], targets ids[1:] shifted,
        with non-assistant positions masked to -100."""
        L = min(block, max(len(e[0]) for e in exs))
        x = torch.zeros((len(exs), L), dtype=torch.long)
        y = torch.full((len(exs), L), -100, dtype=torch.long)
        for i, (ids, mask) in enumerate(exs):
            m_ = min(len(ids) - 1, L)          # predictable positions
            if m_ <= 0:
                continue
            x[i, :m_] = torch.tensor(ids[:m_])
            tgt = torch.tensor(ids[1:m_ + 1])
            mm = torch.tensor(mask[1:m_ + 1])
            y[i, :m_] = tgt.masked_fill(mm == 0, -100)
        return x.to(device), y.to(device)

    rng = np.random.default_rng(7)
    steps_done, losses = 0, []
    t0 = time.time()
    model.train()
    while steps_done < args.steps:
        order = rng.permutation(len(examples))
        for i in range(0, len(order), args.batch_size):
            if steps_done >= args.steps:
                break
            batch = [examples[j] for j in order[i:i + args.batch_size]]
            x, y = make_batch(batch)
            loss = model(x, targets=y).loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            steps_done += 1
            losses.append(loss.item())
            if steps_done % 10 == 0 or steps_done == 1:
                print(f"  online step {steps_done}/{args.steps} | "
                      f"loss {loss.item():.4f}", flush=True)
            if steps_done >= args.steps * len(examples) and steps_done >= args.steps:
                break
        if steps_done >= args.steps * args.epochs and args.epochs > 0 \
                and steps_done >= args.steps:
            break

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    payload = {"model": model.state_dict(), "config": cfg.to_dict(),
               "step": ckpt.get("step", 0), "kind": "online",
               "feedback_examples": len(examples)}
    tmp = args.out + ".tmp"
    torch.save(payload, tmp)
    os.replace(tmp, args.out)
    print(f"[save] {args.out} ({len(examples)} examples, "
          f"{time.time() - t0:.0f}s, mean loss {np.mean(losses):.4f})")

    if os.environ.get("HF_TOKEN"):
        try:
            from huggingface_hub import HfApi
            HfApi(token=os.environ["HF_TOKEN"]).upload_file(
                path_or_fileobj=args.out, path_in_repo="ckpt-online.pt",
                repo_id=args.hf_repo, repo_type="model",
                commit_message=f"online learning: {len(examples)} feedback examples")
            print("[hub ] uploaded ckpt-online.pt")
        except Exception as e:
            print("[hub ] upload failed:", e)


if __name__ == "__main__":
    main()
