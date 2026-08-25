#!/usr/bin/env python3
"""Indus autonomous self-improvement loop.

For each topic the agent:
  1. RESEARCHES the web (Wikipedia API, DuckDuckGo fallback)
  2. REMEMBERS sources in an FTS5 knowledge store (retrieval memory)
  3. LEARNS: eval-gated fine-tune on fresh text + replay buffer, with
     automatic rollback if held-out probe regresses
  4. ANSWERS a grounded question about the topic with citations

Usage:
  python scripts/auto_learn.py --topics "Eiffel Tower,Photosynthesis"
  python scripts/auto_learn.py                 # interactive topic prompt
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indus.autonomous import (KnowledgeStore, SelfLearner, WebCorpus,   # noqa: E402
                              autonomous_cycle)
from indus.config import IndusConfig                                   # noqa: E402
from indus.model import IndusLM, ensure_vocab_size                     # noqa: E402
from indus.tokenizer import BPETokenizer                               # noqa: E402

CKPT_CHAIN = ["checkpoints/ckpt-sft.pt", "checkpoints/ckpt-online.pt",
              "checkpoints/ckpt-final.pt", "checkpoints/ckpt-latest.pt"]


def resolve_ckpt(explicit: str | None, hf_repo: str) -> str:
    if explicit:
        return explicit
    for c in CKPT_CHAIN:
        if os.path.exists(c):
            return c
    from huggingface_hub import hf_hub_download
    return hf_hub_download(hf_repo, "ckpt-latest.pt", repo_type="model")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topics", default=None,
                    help="comma-separated research topics")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--hf-repo", default=os.environ.get("HF_REPO",
                   "AbhijeetJain4075/indus-llm"))
    ap.add_argument("--store", default="knowledge/knowledge.db")
    ap.add_argument("--out", default="checkpoints/ckpt-autonomous.pt")
    ap.add_argument("--steps", type=int, default=60,
                    help="fine-tune steps per topic")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-pages", type=int, default=3)
    ap.add_argument("--no-train", action="store_true",
                    help="research+answer only (no weight updates)")
    ap.add_argument("--upload", action="store_true",
                    help="upload ckpt-autonomous.pt to the HF repo")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = resolve_ckpt(args.ckpt, args.hf_repo)
    print(f"[load] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = IndusConfig.from_dict(ckpt["config"])
    model = IndusLM(cfg).to(device)
    model.load_state_dict({k: v.cpu() for k, v in ckpt["model"].items()},
                          strict=False)

    tok_path = args.tokenizer
    for cand in (tok_path, "data/tokenizer.json", "data_v2/tokenizer.json"):
        if cand and os.path.exists(cand):
            tok_path = cand
            break
    tok = BPETokenizer.load(tok_path)
    tok.add_chat_specials()
    ensure_vocab_size(model, len(tok.vocab))
    model.eval()
    print(f"[model] {cfg.name} | {model.num_params() / 1e6:.2f}M non-emb "
          f"| vocab {len(tok.vocab)} | device {device}")

    store = KnowledgeStore(args.store)
    learner = None if args.no_train else SelfLearner(
        model, tok, device=device,
        replay_bin="data/train.bin" if os.path.exists("data/train.bin")
        else None)

    topics = [t.strip() for t in args.topics.split(",") if t.strip()] \
        if args.topics else []
    while not topics:
        try:
            t = input("\nresearch topic> ").strip()
        except EOFError:
            break
        if not t:
            break
        topics.append(t)

    reports = []
    corpus = WebCorpus(max_pages=args.max_pages)
    for topic in topics:
        print(f"\n{'=' * 60}\nAUTONOMOUS CYCLE: {topic}\n{'=' * 60}")
        rep = autonomous_cycle(model, tok, topic, corpus=corpus, store=store,
                               learner=learner, device=device)
        reports.append(rep)

    learned_any = any(r.get("learning", {}).get("accepted") for r in reports)
    if learned_any and not args.no_train:
        payload = {"model": model.state_dict(), "config": cfg.to_dict(),
                   "step": ckpt.get("step", 0), "kind": "autonomous",
                   "topics": topics}
        tmp = args.out + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, args.out)
        print(f"\n[save] {args.out}")
        if args.upload and os.environ.get("HF_TOKEN"):
            try:
                from huggingface_hub import HfApi
                HfApi(token=os.environ["HF_TOKEN"]).upload_file(
                    path_or_fileobj=args.out, path_in_repo="ckpt-online.pt",
                    repo_id=args.hf_repo, repo_type="model",
                    commit_message=f"autonomous learning: {topics}")
                print("[hub ] uploaded")
            except Exception as e:
                print("[hub ] upload failed:", e)

    os.makedirs("knowledge", exist_ok=True)
    with open("knowledge/last_report.json", "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    print(f"\n[done] report -> knowledge/last_report.json "
          f"({store.count()} docs in store)")


if __name__ == "__main__":
    main()
