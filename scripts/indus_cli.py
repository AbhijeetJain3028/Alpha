#!/usr/bin/env python3
"""Indus Code - an interactive coding/chat CLI for the Indus LLM.

A terminal-native assistant interface: streaming token-by-token replies,
multi-turn memory, sampling controls, checkpoint hot-swapping, transcripts.

Usage:
  python scripts/indus_cli.py                          # auto-pull latest from Hub
  python scripts/indus_cli.py --which ckpt-sft.pt      # chat-tuned weights
  python scripts/indus_cli.py --ckpt local/ckpt.pt     # local checkpoint
"""

import argparse
import datetime
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indus.config import IndusConfig              # noqa: E402
from indus.model import IndusLM                   # noqa: E402
from indus.tokenizer import BPETokenizer          # noqa: E402

BANNER = r"""
   ___  _  _   ___  ,_     _    ___  _    ,  ,
  (' | `_//|  / |)  | \   ||   / |) |)   |\_|
   _) |, //_|_/  |_)|_/ \_||_ /  |_)| \  | \|   C O D E
"""

COLOR = {
    "dim": "\033[2m", "bold": "\033[1m", "cyan": "\033[96m",
    "green": "\033[92m", "yellow": "\033[93m", "red": "\033[91m",
    "magenta": "\033[95m", "reset": "\033[0m",
}


def c(text: str, key: str, enabled: bool) -> str:
    return f"{COLOR[key]}{text}{COLOR['reset']}" if enabled else text


class Session:
    def __init__(self, args):
        self.args = args
        self.device = args.device
        self.tok = None
        self.model = None
        self.cfg = None
        self.ckpt_name = "?"
        self.temperature = args.temperature
        self.top_k = args.top_k
        self.max_tokens = args.max_tokens
        self.system_prompt = ""
        self.history = ""            # rendered conversation prefix
        self.turns = 0
        self.tokens_generated = 0
        self.transcript = []         # (role, text)
        self.load_tokenizer()
        self.load_model(args.ckpt, args.which)

    # ------------------------------------------------------------- loading
    def load_tokenizer(self):
        cand = [args_tok(self.args), "data_v2/tokenizer.json",
                "data/tokenizer.json"]
        for p in cand:
            if p and os.path.exists(p):
                self.tok = BPETokenizer.load(p)
                self.tok.add_chat_specials()
                self.tok_path = p
                return
        raise SystemExit("tokenizer.json not found (pass --tokenizer)")

    def load_model(self, ckpt_path, which):
        if ckpt_path is None:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(self.args.hf_repo, which,
                                        repo_type="model")
            guess = os.path.join(os.path.dirname(ckpt_path), "tokenizer.json")
            if os.path.exists(guess):
                self.tok = BPETokenizer.load(guess)
            self.tok.add_chat_specials() if hasattr(self.tok, "add_chat_specials") else None
            self.ckpt_name = which
        else:
            self.ckpt_name = os.path.basename(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = IndusConfig.from_dict(ckpt["config"])
        if len(self.tok.vocab) != cfg.vocab_size:
            cfg.vocab_size = max(cfg.vocab_size, len(self.tok.vocab))
        self.model = IndusLM(cfg).to(self.device).eval()
        state = {k: v.cpu() for k, v in ckpt["model"].items()}
        missing = [k for k, v in self.model.state_dict().items()
                   if k not in state or state[k].shape != v.shape]
        final = {k: v for k, v in state.items()
                 if k in self.model.state_dict()
                 and self.model.state_dict()[k].shape == v.shape}
        self.model.load_state_dict(final, strict=False)
        self.cfg = cfg
        kind = ckpt.get("kind", "pretrained")
        print(c(f"  loaded {self.ckpt_name} ({kind}, step {ckpt.get('step', '?')}) "
                f"| {self.model.num_params()/1e6:.1f}M params "
                f"| ctx {cfg.block_size}", "dim", self.args.color))
        if missing:
            print(c(f"  note: resized embeddings for chat tokens ({len(missing)} tensors fresh)",
                    "dim", self.args.color))

    # ------------------------------------------------------------ prompting
    def has_chat_template(self) -> bool:
        # tokenizer knows the specials AND the checkpoint's embeddings cover them
        needed = max(self.tok.special_tokens.values()) + 1
        return "<|assistant|>" in self.tok.special_tokens and \
            self.cfg.vocab_size >= needed

    def build_prompt_ids(self, user_msg: str) -> list[int]:
        if self.has_chat_template():
            turn = f"<|user|>\n{user_msg}<|end|>\n<|assistant|>\n"
            prefix = f"<|system|>\n{self.system_prompt}<|end|>\n" \
                if self.system_prompt and self.turns == 0 else ""
            return self.tok.encode_with_specials(prefix + self.history + turn)
        # base model fallback: plain continuation
        return self.tok.encode(self.history + user_msg)

    # ----------------------------------------------------------- generation
    @torch.no_grad()
    def stream_reply(self, ids: list[int]):
        """Yield printable text chunks as tokens are sampled."""
        import codecs
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        x = torch.tensor([ids[-self.cfg.block_size:]], device=self.device)
        eot = self.tok.special_tokens.get("<|endoftext|>")
        end = self.tok.special_tokens.get("<|end|>")
        specials = set(self.tok.special_tokens.values())
        t0 = time.time()
        n_new = 0
        for _ in range(self.max_tokens):
            logits = self.model(x[:, -self.cfg.block_size:]).logits[0, -1]
            if self.temperature <= 1e-6:
                nxt = int(torch.argmax(logits))
            else:
                logits = logits / self.temperature
                if self.top_k:
                    k = min(self.top_k, logits.size(-1))
                    thresh = torch.topk(logits, k).values[-1]
                    logits = logits.masked_fill(logits < thresh, float("-inf"))
                probs = torch.softmax(logits, dim=-1)
                nxt = int(torch.multinomial(probs, 1))
            if nxt in (eot, end):
                break
            n_new += 1
            self.tokens_generated += 1
            if nxt not in specials:
                piece = decoder.decode(self.tok.vocab[nxt])
                if piece:
                    yield piece
            x = torch.cat([x, torch.tensor([[nxt]], device=self.device)], dim=1)
        dur = time.time() - t0
        self.last_rate = n_new / max(dur, 1e-6)

    # ---------------------------------------------------------------- chat
    def reply(self, user_msg: str) -> str:
        ids = self.build_prompt_ids(user_msg)
        chunks = []
        print(c("indus> ", "green", self.args.color), end="", flush=True)
        for piece in self.stream_reply(ids):
            chunks.append(piece)
            print(piece, end="", flush=True)
        print()
        reply_text = "".join(chunks).strip()
        self.history += f"<|user|>\n{user_msg}<|end|>\n<|assistant|>\n{reply_text}<|end|>\n" \
            if self.has_chat_template() else \
            self.history + user_msg + "\n"
        self.turns += 1
        self.transcript += [("you", user_msg), ("indus", reply_text)]
        # remember last exchange for feedback commands
        self.last_prompt, self.last_reply = user_msg, reply_text
        # bounded context: drop oldest half if runaway
        if len(self.history) > self.cfg.block_size * 6:
            self.history = self.history[len(self.history) // 2:]
        return reply_text

    def log_feedback(self, verdict: str, correction: str | None = None) -> None:
        p = getattr(self, "last_prompt", None)
        if not p:
            print(c("  ! no exchange to rate yet", "red", self.args.color))
            return
        os.makedirs("data_feedback", exist_ok=True)
        with open("data_feedback/feedback.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.time(), "session": "cli",
                "prompt": p[:2000], "reply": (self.last_reply or "")[:4000],
                "verdict": verdict, "correction": correction,
            }) + "\n")
        label = {"up": "👍 saved — will reinforce", "down": "👎 saved",
                 "taught": "🎓 taught — trains later"}[verdict]
        print(c(f"  * {label} ({sum(1 for _ in open('data_feedback/feedback.jsonl'))} total)",
                "dim", self.args.color))


HELP = f"""
{COLOR['bold']}commands{COLOR['reset']}
  /help              this help
  /new               clear conversation memory
  /system <text>     set the system prompt
  /temp <float>      sampling temperature (0 = greedy)
  /topk <int>        top-k sampling (0 = off)
  /maxtok <int>      max new tokens per reply
  /model <hub-file>  switch checkpoint from Hub (e.g. ckpt-sft.pt)
  /local <path>      switch to a local .pt checkpoint
  /stats             model & session statistics
  /save [file.md]    save transcript
  /good //bad        rate the last reply (feedback log)
  /teach <answer>    correct the last reply (becomes training data)
  /learn             fine-tune on all feedback now (train_from_feedback)
  /quit              exit
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="local checkpoint path")
    ap.add_argument("--which", default="ckpt-latest.pt",
                    help="Hub file to load when --ckpt is absent")
    ap.add_argument("--hf-repo", default=os.environ.get("HF_REPO",
                   "AbhijeetJain4075/indus-llm"))
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--no-color", action="store_true")
    args = ap.parse_args()
    args.color = not args.no_color and sys.stdout.isatty()

    print(c(BANNER, "cyan", args.color))
    sess = Session(args)
    print(c("  type a message · /help for commands · /quit to exit\n",
            "dim", args.color))

    while True:
        try:
            line = input(c("you> ", "yellow", args.color)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""
            if cmd in ("/quit", "/exit", "/q"):
                break
            elif cmd in ("/help", "/?"):
                print(HELP)
            elif cmd in ("/new", "/clear"):
                sess.history, sess.turns, sess.system_prompt = "", 0, ""
                print(c("  * memory cleared", "dim", args.color))
            elif cmd == "/system":
                sess.system_prompt = arg
                print(c(f"  * system prompt set: {arg!r}", "dim", args.color))
            elif cmd == "/temp":
                sess.temperature = float(arg)
                print(c(f"  * temperature={sess.temperature}", "dim", args.color))
            elif cmd == "/topk":
                sess.top_k = int(arg)
                print(c(f"  * top_k={sess.top_k}", "dim", args.color))
            elif cmd == "/maxtok":
                sess.max_tokens = int(arg)
                print(c(f"  * max_tokens={sess.max_tokens}", "dim", args.color))
            elif cmd == "/model":
                sess.load_model(None, arg)
            elif cmd == "/local":
                if not os.path.exists(arg):
                    print(c("  ! file not found", "red", args.color))
                else:
                    sess.load_model(arg, None)
            elif cmd == "/stats":
                rate = getattr(sess, "last_rate", 0.0)
                print(c(f"  model      : {sess.cfg.name} "
                        f"({sess.model.num_params()/1e6:.1f}M params)", "dim", args.color))
                print(c(f"  checkpoint : {sess.ckpt_name}", "dim", args.color))
                print(c(f"  context    : {sess.cfg.block_size} | vocab {sess.cfg.vocab_size}",
                        "dim", args.color))
                print(c(f"  sampling   : temp {sess.temperature} | top-k {sess.top_k}",
                        "dim", args.color))
                print(c(f"  session    : {sess.turns} turns | "
                        f"{sess.tokens_generated} tokens | last {rate:.1f} tok/s",
                        "dim", args.color))
            elif cmd == "/save":
                fname = arg or f"transcript-{datetime.datetime.now():%Y%m%d-%H%M%S}.md"
                with open(fname, "w") as f:
                    f.write(f"# Indus transcript — {datetime.datetime.now():%Y-%m-%d %H:%M}\n\n")
                    for role, text in sess.transcript:
                        f.write(f"**{'🧑' if role=='you' else '⛰'} {role}:** {text}\n\n")
                print(c(f"  * saved {fname}", "dim", args.color))
            elif cmd in ("/good", "/up"):
                sess.log_feedback("up")
            elif cmd in ("/bad", "/down"):
                sess.log_feedback("down")
            elif cmd == "/teach":
                # /teach <better answer...>  → correction replaces last reply
                if not arg:
                    print(c("  ! usage: /teach <better answer>", "red", args.color))
                else:
                    sess.log_feedback("taught", correction=arg)
            elif cmd == "/learn":
                os.system(f"{sys.executable} "
                          f"{os.path.join(os.path.dirname(__file__), 'train_from_feedback.py')}")
            else:
                print(c("  ! unknown command (/help)", "red", args.color))
            continue

        try:
            sess.reply(line)
        except KeyboardInterrupt:
            print(c("\n  ^C (generation interrupted)", "dim", args.color))

    print(c(f"\n  session: {sess.turns} turns, {sess.tokens_generated} tokens generated. bye!",
            "dim", args.color))


def args_tok(args):
    return args.tokenizer


if __name__ == "__main__":
    main()
