#!/usr/bin/env python3
"""Prepare training data for Indus.

Steps:
  1. Obtain a text corpus (download TinyStories excerpt, or use your own .txt files)
  2. Train the byte-level BPE tokenizer on it
  3. Encode the corpus to a binary token file for fast training reads

Usage:
  # download ~20MB of TinyStories and prepare everything:
  python scripts/prepare_data.py --out data/

  # use your own text files instead:
  python scripts/prepare_data.py --out data/ --input mycorpus/*.txt

Options control vocab size and how much of TinyStories to fetch.
"""

import argparse
import glob
import io
import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indus.tokenizer import BPETokenizer  # noqa: E402

TINYSTORIES_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
                   "resolve/main/TinyStories-train.txt")


def download_tinystories(max_bytes: int) -> str:
    print(f"downloading {max_bytes / 1e6:.0f}MB of TinyStories ...")
    req = urllib.request.Request(TINYSTORIES_URL, headers={"Range": f"bytes=0-{max_bytes - 1}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    # decode then drop any trailing partial story
    text = data.decode("utf-8", errors="ignore")
    marker = "<|endoftext|>"
    last = text.rfind(marker)
    if last != -1:
        text = text[:last]
    print(f"got {len(text) / 1e6:.1f}M characters")
    return text


def load_local_files(patterns: list[str]) -> str:
    texts = []
    n = 0
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path, encoding="utf-8", errors="ignore") as f:
                t = f.read()
                texts.append(t)
                n += len(t)
    if not texts:
        raise FileNotFoundError(f"no files matched: {patterns}")
    print(f"loaded {len(texts)} file(s), {n / 1e6:.1f}M characters")
    return "\n".join(texts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data", help="output directory")
    ap.add_argument("--input", nargs="*", help="glob pattern(s) of local .txt files")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--tinystories-bytes", type=int, default=25_000_000,
                    help="bytes of TinyStories to stream when no --input given")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tok_path = os.path.join(args.out, "tokenizer.json")
    train_bin = os.path.join(args.out, "train.bin")
    val_bin = os.path.join(args.out, "val.bin")

    # ---------------------------------------------------------------- corpus
    if args.input:
        text = load_local_files(args.input)
    else:
        try:
            text = download_tinystories(args.tinystories_bytes)
        except Exception as e:  # offline fallback
            print(f"WARNING: download failed ({e}); using built-in demo corpus.")
            from indus.demo_corpus import DEMO_TEXT
            text = DEMO_TEXT * 50

    split = int(0.98 * len(text))
    train_text, val_text = text[:split], text[split:]

    # ------------------------------------------------------------- tokenizer
    if os.path.exists(tok_path):
        print(f"reusing existing tokenizer at {tok_path}")
        tok = BPETokenizer.load(tok_path)
    else:
        print(f"training BPE tokenizer (vocab_size={args.vocab_size}) "
              f"on {len(train_text):,} chars ...")
        tok = BPETokenizer()
        tok.train(train_text, vocab_size=args.vocab_size, verbose=True)
        tok.save(tok_path)
    print(f"vocab size: {len(tok.vocab)}")

    # ------------------------------------------------------------ encode bins
    for name, chunk in (("train", train_text), ("val", val_text)):
        out_path = train_bin if name == "train" else val_bin
        print(f"encoding {name} split ...")
        ids = tok.encode(chunk)
        import numpy as np
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(out_path)
        with open(out_path + ".meta.json", "w") as f:
            json.dump({"n_tokens": len(arr)}, f)
        print(f"  {out_path}: {len(arr):,} tokens")

    print("\ndone. train with:")
    print("  python scripts/train.py --data-dir data")


if __name__ == "__main__":
    main()
