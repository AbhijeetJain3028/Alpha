#!/usr/bin/env python3
"""Chat / generate text with a trained Indus model.

Examples:
  python scripts/generate.py --ckpt checkpoints/ckpt.pt --prompt "Once upon a time"
  python scripts/generate.py --ckpt checkpoints/ckpt.pt --chat
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indus.config import IndusConfig                    # noqa: E402
from indus.model import IndusLM, ensure_vocab_size      # noqa: E402
from indus.tokenizer import BPETokenizer, ENDOFTEXT     # noqa: E402


def load_model(ckpt_path: str, device: str) -> tuple[IndusLM, IndusConfig]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = IndusConfig.from_dict(ckpt["config"])
    model = IndusLM(cfg).to(device)
    state = {k: v.cpu() for k, v in ckpt["model"].items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    n = model.num_params()
    print(f"loaded {ckpt_path} "
          f"({cfg.name}, step={ckpt.get('step', '?')}, "
          f"{n / 1e6:.2f}M non-emb params, "
          f"{model.num_params(non_embedding=False) / 1e6:.2f}M total)")
    return model, cfg


def sample(model, tok, prompt, device, max_new_tokens, temperature, top_k):
    ids = tok.encode(prompt)
    if not ids:
        ids = [tok.special_tokens[ENDOFTEXT]]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = model.generate(x, max_new_tokens=max_new_tokens,
                       temperature=temperature, top_k=top_k,
                       endoftext_id=tok.special_tokens.get(ENDOFTEXT))
    return tok.decode(y[0].tolist())


def chat_loop(model, tok, device, args):
    eot = tok.special_tokens[ENDOFTEXT]
    has_chat = "<|user|>" in tok.special_tokens
    history = ""
    print("Indus chat (empty line to stop)")
    while True:
        try:
            user = input("\nyou> ").strip()
        except EOFError:
            break
        if not user:
            break
        if has_chat:
            history += f"<|user|>\n{user}<|end|>\n<|assistant|>\n"
            ids = tok.encode_with_specials(history)
            x = torch.tensor([ids], dtype=torch.long, device=device)
            y = model.generate(x, max_new_tokens=args.max_new_tokens,
                               temperature=args.temperature, top_k=args.top_k,
                               endoftext_id=eot)
            gen = y[0].tolist()[len(ids):]
            reply_ids = gen
            # cut at first <|end|> if emitted
            end_id = tok.special_tokens.get("<|end|>")
            if end_id in reply_ids:
                reply_ids = reply_ids[:reply_ids.index(end_id)]
            reply = tok.decode(reply_ids).strip()
            history += f"{reply}<|end|>\n"
            # keep context bounded
            tail_ids = tok.encode_with_specials(history)
            if len(tail_ids) > model.config.block_size - args.max_new_tokens - 8:
                history = history[-(model.config.block_size * 2):]
        else:
            prompt = f"user: {user}\nassistant:"
            ids = tok.encode(prompt)
            x = torch.tensor([ids], dtype=torch.long, device=device)
            y = model.generate(x, max_new_tokens=args.max_new_tokens,
                               temperature=args.temperature, top_k=args.top_k,
                               endoftext_id=eot)
            text = tok.decode(y[0].tolist())
            reply = text[len(tok.decode(ids)):].strip()
        print(f"indus> {reply}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--tokenizer", default=None,
                    help="defaults to data/tokenizer.json next to ckpt's data dir")
    ap.add_argument("--hf-repo", default=os.environ.get("HF_REPO",
                   "AbhijeetJain4075/indus-llm"))
    ap.add_argument("--which", default="ckpt-latest.pt",
                    help="repo file to use with --from-hub (e.g. ckpt-sft.pt)")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--chat", action="store_true", help="interactive chat loop")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    if args.ckpt is None:
        from huggingface_hub import hf_hub_download
        args.ckpt = hf_hub_download(args.hf_repo, args.which, repo_type="model")
        if args.tokenizer is None:
            tok_guess = os.path.join(os.path.dirname(args.ckpt), "tokenizer.json")
            if os.path.exists(tok_guess):
                args.tokenizer = tok_guess
        print(f"[hub ] ckpt: {args.ckpt}")
    model, _ = load_model(args.ckpt, device)

    tok_path = args.tokenizer or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt))),
        "data_v2", "tokenizer.json")
    for cand in (tok_path, "data/tokenizer.json", "data_v2/tokenizer.json"):
        if os.path.exists(cand):
            tok_path = cand
            break
    tok = BPETokenizer.load(tok_path)
    tok.add_chat_specials()          # parity with webapp; idempotent
    ensure_vocab_size(model, len(tok.vocab))   # deterministic EOT-copy rows
    print(f"tokenizer: {tok_path} | vocab {len(tok.vocab)} "
          f"| chat {'on' if '<|assistant|>' in tok.special_tokens else 'off'}")

    if args.chat:
        chat_loop(model, tok, device, args)
    else:
        out = sample(model, tok, args.prompt, device,
                     args.max_new_tokens, args.temperature, args.top_k)
        print(out)


if __name__ == "__main__":
    main()
