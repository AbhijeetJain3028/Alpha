#!/usr/bin/env python3
"""Build Indus SFT (chat) dataset from license-checked human-written sources.

Sources:
  OpenAssistant/oasst1            Apache-2.0     multi-turn conversations (en)
  databricks-dolly-15k            CC-BY-SA-3.0   instruction/response pairs
  HuggingFaceTB/smoltalk          Apache-2.0     everyday-conversations slice

Format (specials registered by indus.tokenizer.BPETokenizer.CHAT_SPECIALS):
    <|user|>\n{msg}<|end|>\n<|assistant|>\n{msg}<|end|>\n ...

Outputs data_sft/sft_train.bin (token ids) + sft_mask.bin (uint8: 1 = train on
this token, i.e. assistant content incl. its <|end|>).
"""

import argparse
import gzip
import json
import os
import sys
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indus.tokenizer import BPETokenizer  # noqa: E402

OASST_URL = ("https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/"
             "2023-04-12_oasst_all.messages.jsonl.gz")
DOLLY_URL = ("https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
             "resolve/main/databricks-dolly-15k.jsonl")
SMOLTALK_DIR = ("https://huggingface.co/datasets/HuggingFaceTB/smoltalk/"
                "resolve/main/data/everyday-conversations")


def _dl_bytes(url: str, dest: str) -> bool:
    print(f"[get ] {url.rsplit('/', 1)[-1]}")
    tmp = dest + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as e:
        print(f"[warn] download failed, skipping source: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    os.replace(tmp, dest)
    return True


# ------------------------------------------------------------------- oasst1
def _label_value(m: dict, name: str, default=0):
    lab = m.get("labels") or {}
    v = lab.get(name)
    if isinstance(v, dict):
        return v.get("value", default)
    return v if v is not None else default


def oasst1_conversations(path: str, max_conv: int) -> list[list[dict]]:
    msgs = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            if (m.get("lang") == "en" and m.get("review_result")
                    and not m.get("deleted")):
                msgs[m["message_id"]] = m
    # children map
    kids: dict[str, list[dict]] = {}
    roots = []
    for m in msgs.values():
        p = m.get("parent_id")
        if p is None:
            roots.append(m)
        elif p in msgs:
            kids.setdefault(p, []).append(m)

    def role_of(m):
        return "assistant" if m.get("role") == "assistant" else "user"

    convos = []
    for root in roots:
        seq = [root]
        cur = root
        while len(kids.get(cur["message_id"], [])) > 0 and len(seq) < 8:
            children = kids[cur["message_id"]]
            if cur["role"] == "assistant":
                nxt = min((c for c in children
                           if c.get("rank") is not None),
                          key=lambda c: c["rank"], default=None)
            else:
                nxt = max(children,
                          key=lambda c: float(_label_value(c, "quality")),
                          default=None)
            if nxt is None:
                break
            seq.append(nxt)
            cur = nxt
        # must end on an assistant turn to be a usable example
        if len(seq) >= 2 and seq[-1]["role"] == "assistant":
            convos.append([{"role": role_of(m), "content": m["text"]}
                           for m in seq])
        if len(convos) >= max_conv:
            break
    return convos


def dolly_conversations(path: str, max_conv: int) -> list[list[dict]]:
    convos = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            user = r["instruction"].strip()
            if r.get("context", "").strip():
                user += "\n\nContext:\n" + r["context"].strip()
            convos.append([{"role": "user", "content": user},
                           {"role": "assistant", "content": r["response"].strip()}])
            if len(convos) >= max_conv:
                break
    return convos


def smoltalk_conversations(parquet_dir: str, max_conv: int) -> list[list[dict]]:
    import glob
    import pyarrow.parquet as pq
    convos = []
    for pf_path in sorted(glob.glob(os.path.join(parquet_dir, "*.parquet"))):
        pf = pq.ParquetFile(pf_path)
        for batch in pf.iter_batches(batch_size=256, columns=["messages"]):
            for messages in batch.column("messages").to_pylist():
                conv = [{"role": m["role"], "content": m["content"]}
                        for m in messages]
                convos.append(conv)
                if len(convos) >= max_conv:
                    return convos
    return convos


# ------------------------------------------------------------------ encoding
def render_segments(conv: list[dict]) -> list[tuple[str, bool]]:
    """Return [(text, train_on_it)] segments; specials included in text."""
    segs = []
    for m in conv:
        role = m["role"]
        tag = f"<|{role}|>"
        segs.append((f"{tag}\n", False))
        segs.append((m["content"].strip(), role == "assistant"))
        segs.append(("<|end|>\n", role == "assistant"))
    return segs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", default="data_v2/tokenizer.json")
    ap.add_argument("--out-dir", default="data_sft")
    ap.add_argument("--max-oasst", type=int, default=8000)
    ap.add_argument("--max-dolly", type=int, default=15011)
    ap.add_argument("--max-smoltalk", type=int, default=2000)
    ap.add_argument("--val-permille", type=int, default=20)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("corpus", exist_ok=True)

    oasst_gz = "corpus/oasst1_all.jsonl.gz"
    have_oasst = os.path.exists(oasst_gz) or \
        _dl_bytes(OASST_URL, oasst_gz)
    dolly_jsonl = "corpus/dolly15k.jsonl"
    have_dolly = os.path.exists(dolly_jsonl) or \
        _dl_bytes(DOLLY_URL, dolly_jsonl)
    st_dir = os.path.join("corpus", "smoltalk_everyday")
    if not os.path.exists(st_dir):
        os.makedirs(st_dir, exist_ok=True)
        for name in ["train-00000-of-00001.parquet"]:
            _dl_bytes(f"{SMOLTALK_DIR}/{name}", os.path.join(st_dir, name))
    have_st = any(f.endswith(".parquet") and
                  os.path.getsize(os.path.join(st_dir, f)) > 0
                  for f in os.listdir(st_dir)) if os.path.isdir(st_dir) else False

    print("[data] extracting conversations ...")
    convos = []
    if have_oasst:
        convos += [("oasst1", c) for c in
                   oasst1_conversations(oasst_gz, args.max_oasst)]
    if have_dolly:
        convos += [("dolly", c) for c in
                   dolly_conversations(dolly_jsonl, args.max_dolly)]
    if have_st:
        convos += [("smoltalk", c) for c in
                   smoltalk_conversations(st_dir, args.max_smoltalk)]
    if not convos:
        raise SystemExit("no SFT sources available - all downloads failed")
    from collections import Counter
    print("  source counts:", dict(Counter(s for s, _ in convos)),
          "| total:", len(convos))

    tok = BPETokenizer.load(args.tokenizer)
    tok.add_chat_specials()
    tok.save(args.tokenizer)
    sid = tok.special_tokens
    eot_id = sid["<|endoftext|>"]

    def encode_conv(segs):
        ids, mask = [], []
        for text, train_on in segs:
            if text in tok.special_tokens:
                ids.append(sid[text])
                mask.append(1 if train_on else 0)
            else:
                piece = tok.encode(text)
                ids.extend(piece)
                mask.extend([1 if train_on else 0] * len(piece))
        return ids, mask

    import hashlib
    tr_i, tr_m, va_i, va_m = [], [], [], []

    def emit(ids, mask):
        h = int(hashlib.md5(str(ids[:64]).encode()).hexdigest(), 16) % 1000
        if h < args.val_permille:
            va_i.extend(ids)
            va_m.extend(mask)
        else:
            tr_i.extend(ids)
            tr_m.extend(mask)

    n_tok = 0
    for src, conv in convos:
        ids, mask = encode_conv(render_segments(conv))
        ids.append(eot_id)
        mask.append(0)
        n_tok += len(ids)
        emit(ids, mask)

    np.array(tr_i, dtype=np.uint16).tofile(os.path.join(args.out_dir, "sft_train.bin"))
    np.array(tr_m, dtype=np.uint8).tofile(os.path.join(args.out_dir, "sft_train.mask.bin"))
    np.array(va_i, dtype=np.uint16).tofile(os.path.join(args.out_dir, "sft_val.bin"))
    np.array(va_m, dtype=np.uint8).tofile(os.path.join(args.out_dir, "sft_val.mask.bin"))

    meta = {
        "conversations": len(convos),
        "tokens": n_tok,
        "assistant_token_fraction": round(sum(tr_m) / max(len(tr_m), 1), 3),
        "vocab_size": len(tok.vocab),
        "sources": {"oasst1": "Apache-2.0", "dolly": "CC-BY-SA-3.0",
                    "smoltalk": "Apache-2.0"},
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] {meta}")


if __name__ == "__main__":
    main()
