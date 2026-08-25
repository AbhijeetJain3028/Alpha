#!/usr/bin/env python3
"""Build Indus pretraining corpus v2 from real, license-checked open sources.

Sources (licenses recorded in corpus/manifest.json):
  TinyStories   CDLA-Sharing-1.0   narrative fluency
  FineWeb-Edu   ODC-BY-1.0         educational web text (one sample parquet)
  WikiText-103  CC BY-SA 3.0       factual / encyclopedic text

Pipeline:
  1. download + normalize each source into corpus/<source>.txt, documents
     separated by <|endoftext|> lines
  2. train BPE tokenizer v2 on a stratified sample of the mix
  3. stream-encode every document (real EOT ids between docs) into
     data_v2/train.bin + val.bin using a hash-based 98/2 doc split

RAM stays bounded: only a few MB are held at any moment.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indus.tokenizer import BPETokenizer, ENDOFTEXT  # noqa: E402

CORPUS = "corpus"
EOT = ENDOFTEXT
SEP = "\n" + EOT + "\n"
WORD_RE = re.compile(r"\s*\S+|\s+")

TINYSTORIES_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/"
                   "resolve/main/TinyStories-train.txt")
FINEWEB_DIR = ("https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/"
               "resolve/main/sample/10BT")
WIKITEXT_URLS = [
    ("https://huggingface.co/datasets/Salesforce/wikitext/"
     "resolve/main/wikitext-103-raw-v1/train-00000-of-00002.parquet"),
    ("https://huggingface.co/datasets/Salesforce/wikitext/"
     "resolve/main/wikitext-103-raw-v1/train-00001-of-00002.parquet"),
]
LICENSES = {
    "tinystories": "CDLA-Sharing-1.0",
    "fineweb-edu": "ODC-BY-1.0",
    "wikitext103": "CC-BY-SA-3.0",
}
SOURCES = ["tinystories", "fineweb-edu", "wikitext103"]


# ------------------------------------------------------------------ fetchers
def _dl(url: str, dest: str, headers: dict | None = None) -> None:
    print(f"[get ] {url.rsplit('/', 1)[-1]}", flush=True)
    req = urllib.request.Request(url, headers=headers or {})
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
        while True:
            block = r.read(1 << 20)
            if not block:
                break
            f.write(block)
    os.replace(tmp, dest)


def _flush(path_fh, buf: list[str], min_len: int) -> int:
    n = 0
    for d in buf:
        d = d.strip()
        if len(d) >= min_len:
            path_fh.write(d + SEP)
            n += 1
    return n


def fetch_tinystories(max_bytes: int) -> str:
    out = os.path.join(CORPUS, "tinystories.txt")
    if not os.path.exists(out):
        req = urllib.request.Request(TINYSTORIES_URL,
                                     headers={"Range": f"bytes=0-{max_bytes - 1}"})
        print(f"[get ] TinyStories ({max_bytes / 1e6:.0f}MB range)", flush=True)
        n_docs, buf = 0, []
        with urllib.request.urlopen(req, timeout=600) as r, \
                open(out + ".tmp", "w", encoding="utf-8") as fh:
            carry = ""
            while True:
                block = r.read(1 << 20).decode("utf-8", errors="ignore")
                if not block:
                    break
                carry += block
                parts = carry.split(EOT)
                carry = parts.pop()
                buf.extend(parts)
                if len(buf) >= 4096:
                    n_docs += _flush(fh, buf, 40)
                    buf.clear()
        with open(out + ".tmp", "a", encoding="utf-8") as fh:
            n_docs += _flush(fh, [carry], 40)
        os.replace(out + ".tmp", out)
        print(f"[ok  ] tinystories: {n_docs:,} docs", flush=True)
    return out


def fetch_fineweb_edu(file_idx: int) -> str:
    out = os.path.join(CORPUS, "fineweb-edu.txt")
    if not os.path.exists(out):
        import pyarrow.parquet as pq
        url = f"{FINEWEB_DIR}/{file_idx:03d}_00000.parquet"
        local = os.path.join(CORPUS, url.rsplit("/", 1)[-1])
        if not os.path.exists(local):
            _dl(url, local)
        pf = pq.ParquetFile(local)
        n_docs, buf = 0, []
        with open(out + ".tmp", "w", encoding="utf-8") as fh:
            for batch in pf.iter_batches(batch_size=2048, columns=["text"]):
                buf.extend(t for t in batch.column("text").to_pylist())
                if len(buf) >= 2048:
                    n_docs += _flush(fh, buf, 200)
                    buf.clear()
            n_docs += _flush(fh, buf, 200)
        os.remove(local)
        os.replace(out + ".tmp", out)
        print(f"[ok  ] fineweb-edu: {n_docs:,} docs", flush=True)
    return out


def fetch_wikitext() -> str:
    out = os.path.join(CORPUS, "wikitext103.txt")
    if not os.path.exists(out):
        import pyarrow.parquet as pq
        shards = []
        for i, u in enumerate(WIKITEXT_URLS):
            shard = os.path.join(CORPUS, f"wt103_{i}.parquet")
            if not os.path.exists(shard):
                _dl(u, shard)
            shards.append(shard)
        n_docs, article = 0, []

        def close_article():
            nonlocal article
            text = "".join(article).strip()
            article = []
            return text

        with open(out + ".tmp", "w", encoding="utf-8") as fh:
            for shard in shards:
                pf = pq.ParquetFile(shard)
                for batch in pf.iter_batches(batch_size=8192, columns=["text"]):
                    for line in batch.column("text").to_pylist():
                        stripped = line.strip()
                        # wiki article headings look like "= Title ="
                        if stripped.startswith("= ") and stripped.endswith(" ="):
                            text = close_article()
                            if len(text) >= 200:
                                fh.write(text + SEP)
                                n_docs += 1
                                if n_docs % 2048 == 0:
                                    fh.flush()
                        article.append(line)
                    if len(article) > 400_000:      # runaway page guard
                        text = close_article()
                        if len(text) >= 200:
                            fh.write(text + SEP)
                            n_docs += 1
                text = close_article()              # flush at shard boundary
                if len(text) >= 200:
                    fh.write(text + SEP)
                    n_docs += 1
        for shard in shards:
            os.remove(shard)
        os.replace(out + ".tmp", out)
        print(f"[ok  ] wikitext103: {n_docs:,} docs", flush=True)
    return out


# ------------------------------------------------------------- streaming utils
def iter_docs(path: str):
    """Stream documents from a normalized corpus file (bounded RAM)."""
    with open(path, encoding="utf-8", buffering=1 << 22) as f:
        carry = ""
        while True:
            block = f.read(1 << 23)             # 8MB
            if not block:
                break
            data = carry + block
            parts = data.split(SEP)
            carry = parts.pop()
            for doc in parts:
                if doc.strip():
                    yield doc
        if carry.strip():
            yield carry


def source_stats(sources: dict) -> dict:
    stats = {}
    for src in SOURCES:
        chars = docs = 0
        for doc in iter_docs(sources[src]):
            chars += len(doc) + len(SEP)
            docs += 1
        stats[src] = {"chars": chars, "docs": docs, "license": LICENSES[src]}
    return stats


def iter_chunks(sources: dict, chunk_chars: int):
    """Yield ~chunk_chars chunks assembled doc-by-doc across all sources."""
    buf, blen = [], 0
    for src in SOURCES:
        for doc in iter_docs(sources[src]):
            buf.append(doc)
            blen += len(doc)
            if blen >= chunk_chars:
                yield SEP.join(buf)
                buf, blen = [], 0
    if buf:
        yield SEP.join(buf)


# ------------------------------------------------------------------- encoding
_WORKER_TOK = None


def _init_worker(tok_path: str):
    global _WORKER_TOK
    _WORKER_TOK = BPETokenizer.load(tok_path)


def _encode_chunk(chunk: str, val_permille: int):
    tok = _WORKER_TOK
    eot_id = tok.special_tokens[ENDOFTEXT]
    tr, va = [], []
    memo: dict[str, list[int]] = {}
    finditer = WORD_RE.finditer
    for doc in chunk.split(SEP):
        if not doc.strip():
            continue
        h = int(hashlib.md5(doc[:256].encode("utf-8")).hexdigest(), 16) % 1000
        target = va if h < val_permille else tr
        for m in finditer(doc):
            w = m.group()
            got = memo.get(w)
            if got is None:
                got = tok._encode_word(list(w.encode("utf-8")))
                memo[w] = got
            target.extend(got)
        target.append(eot_id)
    return np.array(tr, dtype=np.uint16), np.array(va, dtype=np.uint16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ts-bytes", type=int, default=500_000_000,
                    help="bytes of TinyStories to stream")
    ap.add_argument("--fineweb-file", type=int, default=0)
    ap.add_argument("--vocab-size", type=int, default=16384)
    ap.add_argument("--tokenizer-sample-chars", type=int, default=120_000_000)
    ap.add_argument("--val-permille", type=int, default=20)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--chunk-chars", type=int, default=400_000)
    args = ap.parse_args()

    os.makedirs(CORPUS, exist_ok=True)

    paths = {
        "tinystories": fetch_tinystories(args.ts_bytes),
        "fineweb-edu": fetch_fineweb_edu(args.fineweb_file),
        "wikitext103": fetch_wikitext(),
    }
    sources = {k: v for k, v in paths.items()}

    print("[stat] counting corpus ...", flush=True)
    stats = source_stats(sources)
    total_chars = sum(s["chars"] for s in stats.values())
    for src, s in stats.items():
        print(f"  {src:14s} {s['docs']:>9,} docs {s['chars'] / 1e6:8.1f}M chars "
              f"({s['license']})")
    print(f"  {'TOTAL':14s} {'':>12}{total_chars / 1e6:8.1f}M chars")
    with open(os.path.join(CORPUS, "manifest.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # ------------------------------------------------------------- tokenizer
    tok_path = "data_v2/tokenizer.json"
    os.makedirs("data_v2", exist_ok=True)
    if os.path.exists(tok_path):
        print(f"[tok ] exists: {tok_path}")
    else:
        print(f"[tok ] training BPE v2 vocab={args.vocab_size} on stratified sample")
        parts = []
        for src in SOURCES:
            share = stats[src]["chars"] / max(total_chars, 1)
            budget = args.tokenizer_sample_chars * share
            step = max(1, int(stats[src]["chars"] / max(budget, 1)))
            taken, pos = [], 0
            for doc in iter_docs(sources[src]):
                if pos % step == 0 and sum(len(d) for d in taken[-50:]) < budget:
                    taken.append(doc)
                pos += 1
            chars_taken = sum(len(d) for d in taken)
            print(f"  {src}: {len(taken):,} docs ({chars_taken / 1e6:.1f}M chars)")
            parts.extend(taken)
        sample = SEP.join(parts)
        del parts
        print(f"  sample total: {len(sample) / 1e6:.1f}M chars")
        tok = BPETokenizer()
        t0 = time.time()
        tok.train(sample, vocab_size=args.vocab_size, verbose=True,
                  progress_file="data_v2/.tokenizer_progress.json")
        print(f"  tokenizer trained in {time.time() - t0:.0f}s")
        tok.save(tok_path)
        del sample
        if os.path.exists("data_v2/.tokenizer_progress.json"):
            os.remove("data_v2/.tokenizer_progress.json")
    tok_meta = BPETokenizer.load(tok_path)
    print(f"[tok ] vocab size: {len(tok_meta.vocab)}")

    # ---------------------------------------------------------------- encode
    # resumable: progress file records chunk count + exact byte offsets,
    # bins are truncated back to those offsets before appending again
    prog_path = "data_v2/.encode_progress.json"
    start_n = tr_bytes = va_bytes = 0
    if os.path.exists(prog_path):
        with open(prog_path) as f:
            p = json.load(f)
        start_n, tr_bytes, va_bytes = p["chunks"], p["train_bytes"], p["val_bytes"]
        if (os.path.getsize("data_v2/train.bin") < tr_bytes or
                os.path.getsize("data_v2/val.bin") < va_bytes):
            start_n = 0                       # inconsistent - restart clean
    print(f"[enc ] workers={args.workers} chunk~{args.chunk_chars // 1000}k chars "
          f"| resuming at chunk {start_n}", flush=True)

    import itertools
    mode = "ab" if start_n else "wb"
    train_f = open("data_v2/train.bin", mode)
    val_f = open("data_v2/val.bin", mode)
    if start_n:
        os.truncate("data_v2/train.bin", tr_bytes)
        os.truncate("data_v2/val.bin", va_bytes)
    n_tr = n_va = 0
    n_chunks = start_n
    t0 = time.time()
    window: deque = deque()

    def drain_one(ex_done=False):
        nonlocal n_tr, n_va, n_chunks, tr_bytes, va_bytes
        tr, va = window.popleft().result()
        tr.tofile(train_f)
        va.tofile(val_f)
        n_tr += len(tr)
        n_va += len(va)
        n_chunks += 1
        if n_chunks % 10 == 0 or ex_done:
            train_f.flush()
            val_f.flush()
            os.sync()
            tr_bytes = os.path.getsize("data_v2/train.bin")
            va_bytes = os.path.getsize("data_v2/val.bin")
            with open(prog_path, "w") as f:
                json.dump({"chunks": n_chunks, "train_bytes": tr_bytes,
                           "val_bytes": va_bytes}, f)
        if n_chunks % 25 == 0 or ex_done:
            el = time.time() - t0
            rate = (n_tr + n_va) / max(el, 1e-6)
            print(f"  {n_chunks} chunks | {n_tr + n_va:,} tok | {rate / 1e3:.0f}k tok/s",
                  flush=True)

    with ProcessPoolExecutor(args.workers, initializer=_init_worker,
                             initargs=(tok_path,)) as ex:
        max_pending = args.workers * 3
        chunks = iter_chunks(sources, args.chunk_chars)
        if start_n:
            chunks = itertools.islice(chunks, start_n, None)
        for chunk in chunks:
            window.append(ex.submit(_encode_chunk, chunk, args.val_permille))
            if len(window) >= max_pending:
                drain_one()
        while window:
            drain_one(ex_done=(len(window) == 1))
    train_f.close()
    val_f.close()
    if os.path.exists(prog_path):
        os.remove(prog_path)

    for name, n in (("train", n_tr), ("val", n_va)):
        with open(f"data_v2/{name}.bin.meta.json", "w") as f:
            json.dump({"n_tokens": n}, f)
    ratio = total_chars / max(n_tr + n_va, 1)
    print(f"\n[done] train={n_tr:,} val={n_va:,} tokens | "
          f"{ratio:.2f} chars/token | {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
