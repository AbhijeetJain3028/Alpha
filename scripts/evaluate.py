#!/usr/bin/env python3
"""Evaluate an Indus checkpoint on real benchmarks.

Metrics:
  - WikiText-103 validation perplexity (tokenized with Indus BPE)
  - LAMBADA (OpenAI split) zero-shot last-word accuracy
  - ARC-Easy zero-shot accuracy (loglikelihood ranking over choices)

Usage:
  python scripts/evaluate.py --ckpt ckpt-latest.pt --tokenizer data_v2/tokenizer.json
  python scripts/evaluate.py --hf-repo AbhijeetJain4075/indus-llm --from-hub
"""

import argparse
import json
import math
import os
import sys
import urllib.request

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indus.config import IndusConfig            # noqa: E402
from indus.model import IndusLM                 # noqa: E402
from indus.tokenizer import BPETokenizer        # noqa: E402

WIKITEXT_VAL = ("https://huggingface.co/datasets/Salesforce/wikitext/"
                "resolve/main/wikitext-103-raw-v1/validation-00000-of-00001.parquet")
LAMBADA_URLS = [
    "https://huggingface.co/datasets/EleutherAI/lambada_openai/resolve/main/"
    "data/lambada_test_en.jsonl",
    "https://huggingface.co/datasets/EleutherAI/lambada_openai/resolve/main/"
    "lambada_openai.txt",
]
ARC_EASY = ("https://huggingface.co/datasets/allenai/ai2_arc/"
            "resolve/main/ARC-Easy/test-00000-of-00001.parquet")


def _dl(url: str, dest: str) -> str:
    if os.path.exists(dest):
        return dest
    print(f"[get ] {url.rsplit('/', 1)[-1]}")
    tmp = dest + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
        os.replace(tmp, dest)
        return dest
    except Exception as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        print(f"[warn] download failed: {e}")
        return None


# ------------------------------------------------------------------- metrics
@torch.no_grad()
def wikitext_ppl(model, tok, device, max_chars=4_000_000):
    """Perplexity on WikiText-103 validation text, non-overlapping windows."""
    import pyarrow.parquet as pq
    p = _dl(WIKITEXT_VAL, "corpus/wikitext103_val.parquet")
    if not p:
        return None
    pf = pq.ParquetFile(p)
    texts = []
    remaining = max_chars
    for batch in pf.iter_batches(batch_size=8192, columns=["text"]):
        for line in batch.column("text").to_pylist():
            texts.append(line)
            remaining -= len(line)
            if remaining <= 0:
                break
        if remaining <= 0:
            break
    text = "".join(texts)
    del texts
    ids = tok.encode(text)
    block = model.config.block_size
    steps = max(0, len(ids) - 1) // block
    if steps == 0:
        return None
    nll, count = 0.0, 0
    for i in range(steps):
        x = torch.tensor([ids[i * block:(i + 1) * block]], device=device)
        y = torch.tensor([ids[i * block + 1:(i + 1) * block + 1]], device=device)
        logits = model(x).logits
        nll += torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1),
            reduction="sum").item()
        count += block
    return math.exp(nll / count), count


@torch.no_grad()
def lambada_acc(model, tok, device, max_examples=1500):
    """Zero-shot: predict final word given long context; exact word match."""
    path = None
    for u in LAMBADA_URLS:
        path = _dl(u, "corpus/lambada_test_en.txt")
        if path:
            break
    if not path:
        return None
    correct = total = 0
    buffer = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            if path.endswith(".jsonl"):            # new HF layout
                try:
                    line = json.loads(line).get("text", "")
                except json.JSONDecodeError:
                    pass
            buffer.append(line.strip())
            if len(buffer) >= max_examples:
                break
    for text in buffer:
        if not text or " " not in text.strip():
            continue
        ctx, target_word = text.rsplit(" ", 1)
        target_ids = tok.encode(" " + target_word)
        ctx_ids = tok.encode(ctx)[-(model.config.block_size -
                                    len(target_ids)):]
        x = torch.tensor([ctx_ids], device=device)
        logits = model(x).logits[0, -1]          # next-token distribution
        # teacher-forced greedy match across target tokens
        hit = True
        cur = x
        for t in target_ids:
            nxt = torch.argmax(logits[-1]).item()
            if nxt != t:
                hit = False
                break
            cur = torch.cat([cur, torch.tensor([[t]], device=device)], dim=1)
            logits = model(cur).logits[0, -1]
        correct += int(hit)
        total += 1
    return correct / max(total, 1), total


@torch.no_grad()
def arc_easy_acc(model, tok, device, max_examples=1000):
    """Zero-shot ARC-Easy: rank answer options by summed token logprob."""
    import pyarrow.parquet as pq
    p = _dl(ARC_EASY, "corpus/arc_easy_test.parquet")
    if not p:
        return None
    pf = pq.ParquetFile(p)
    rows = []
    for batch in pf.iter_batches(batch_size=512,
                                 columns=["question", "choices", "answerKey"]):
        qs = batch.column("question").to_pylist()
        chs = batch.column("choices").to_pylist()
        ks = batch.column("answerKey").to_pylist()
        rows.extend(zip(qs, chs, ks))
        if len(rows) >= max_examples:
            break

    def seq_logprob(prompt_ids, cont_ids):
        # teacher-forced LM scoring: input = ids[:-1], targets = ids[1:]
        ids = (prompt_ids + cont_ids)[-(model.config.block_size + 1):]
        L = len(cont_ids)
        if len(ids) < L + 1:
            return -1e9
        x = torch.tensor([ids[:-1]], device=device)
        y = torch.tensor([ids[1:]], device=device)
        logits = model(x).logits
        lp = torch.log_softmax(logits[:, -L:].float(), dim=-1)
        return lp[0, torch.arange(L), y[0][-L:]].sum().item()

    correct = total = 0
    for q, choices, key in rows:
        labels = choices["label"]
        texts = choices["text"]
        if key not in labels:
            continue
        gold = labels.index(key)
        prompt = tok.encode(f"Question: {q}\nAnswer:")
        scores = [seq_logprob(prompt, tok.encode(" " + t)) for t in texts]
        correct += int(int(np.argmax(scores)) == gold)
        total += 1
    return correct / max(total, 1), total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tokenizer", default="data_v2/tokenizer.json")
    ap.add_argument("--hf-repo", default=os.environ.get("HF_REPO",
                   "AbhijeetJain4075/indus-llm"))
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-lambada", type=int, default=1500)
    ap.add_argument("--max-arc", type=int, default=1000)
    args = ap.parse_args()

    if args.ckpt is None:
        from huggingface_hub import hf_hub_download
        args.ckpt = hf_hub_download(args.hf_repo, "ckpt-latest.pt",
                                    repo_type="model")
        print(f"[hub ] {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = IndusConfig.from_dict(ckpt["config"])
    model = IndusLM(cfg)
    model.load_state_dict({k: v.cpu() for k, v in ckpt["model"].items()})
    model = model.to(args.device).eval()
    step = ckpt.get("step", "?")

    tok = BPETokenizer.load(args.tokenizer)
    if len(tok.vocab) != cfg.vocab_size:
        alt = "data/tokenizer.json"
        print(f"[warn] tokenizer vocab {len(tok.vocab)} != ckpt "
              f"vocab {cfg.vocab_size}")
        if os.path.exists(alt) and \
                len(BPETokenizer.load(alt).vocab) == cfg.vocab_size:
            args.tokenizer = alt
            tok = BPETokenizer.load(alt)
            print(f"[warn] switched to matching tokenizer: {alt}")

    results = {"checkpoint_step": step, "params_M":
               round(model.num_params() / 1e6, 2)}

    r = wikitext_ppl(model, tok, args.device)
    if r:
        results["wikitext103_ppl"] = round(r[0], 2)
        results["wikitext_tokens"] = r[1]
        print(f"WikiText-103 PPL: {r[0]:.2f} ({r[1]:,} tokens)")

    r = lambada_acc(model, tok, args.device, args.max_lambada)
    if r:
        results["lambada_acc"] = round(r[0], 4)
        results["lambada_n"] = r[1]
        print(f"LAMBADA acc: {r[0] * 100:.2f}% (n={r[1]})")

    r = arc_easy_acc(model, tok, args.device, args.max_arc)
    if r:
        results["arc_easy_acc"] = round(r[0], 4)
        results["arc_easy_n"] = r[1]
        print(f"ARC-Easy acc: {r[0] * 100:.2f}% (n={r[1]})")

    out = "eval_results.json"
    existing = {}
    if os.path.exists(out):
        with open(out) as f:
            existing = json.load(f)
    existing[str(step)] = results
    with open(out, "w") as f:
        json.dump(existing, f, indent=2)
    print("\n", json.dumps(results, indent=2))

    if os.environ.get("HF_TOKEN") and args.hf_repo:
        try:
            from huggingface_hub import HfApi
            HfApi(token=os.environ["HF_TOKEN"]).upload_file(
                path_or_fileobj=out, path_in_repo="eval_results.json",
                repo_id=args.hf_repo, repo_type="model",
                commit_message=f"Eval at step {step}")
            print("[hub ] uploaded eval_results.json")
        except Exception as e:
            print("[hub ] upload failed:", e)


if __name__ == "__main__":
    main()
