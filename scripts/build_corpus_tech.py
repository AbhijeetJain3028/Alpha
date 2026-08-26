#!/usr/bin/env python3
"""Build Indus TECHNICAL corpus (GitHub/HF/Kaggle-derived, license-checked).

Fetches + normalizes tech-industry sources into SEP-separated doc files and
emits a sources JSON consumable by build_corpus.py --sources-json:

  Source            Origin                    License
  stackedu_py       GitHub->Software Heritage (Stack-Edu terms, permissive-derived)
  stackedu_md       GitHub READMEs/docs       (Stack-Edu terms)
  nemotron_code     NVIDIA synthetic          CC-BY-4.0
  nemotron_stem     NVIDIA synthetic          CC-BY-4.0
  fineweb_edu       web (glue language)       ODC-BY-1.0

Usage:
  python scripts/build_corpus_tech.py --shards 1
  python scripts/build_corpus.py --sources-json corpus_tech/sources.json \
      --tokenizer data_v3/tokenizer.json
"""

import argparse
import importlib.util
import json
import os
import sys

import pyarrow.parquet as pq
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# reuse SEP + iter_docs-style normalization from the main builder
spec = importlib.util.spec_from_file_location(
    "bc", os.path.join(ROOT, "scripts", "build_corpus.py"))
_bc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_bc)
SEP = _bc.SEP

HF = "https://huggingface.co/datasets"
SOURCES_URL = {
    "stackedu_py": f"{HF}/HuggingFaceTB/stack-edu/resolve/main/Python/train-0000{i}-of-00005.parquet",
    "stackedu_md": f"{HF}/HuggingFaceTB/stack-edu/resolve/main/Markdown/train-0000{i}-of-00005.parquet",
    "nemotron_code": f"{HF}/nvidia/Nemotron-Post-Training-Dataset-v2/resolve/main/data/code-0000{i}-of-00002.parquet",
    "nemotron_stem": f"{HF}/nvidia/Nemotron-Post-Training-Dataset-v2/resolve/main/data/stem-0000{i}-of-00002.parquet",
}
LICENSES = {
    "stackedu_py": "Stack-Edu terms (Stack-v2, permissive-derived)",
    "stackedu_md": "Stack-Edu terms (Stack-v2, permissive-derived)",
    "nemotron_code": "CC-BY-4.0",
    "nemotron_stem": "CC-BY-4.0",
    "fineweb_edu": "ODC-BY-1.0",
}


def _dl(url: str, dest: str) -> bool:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return True
    print(f"[get ] {os.path.basename(dest)}", flush=True)
    tmp = dest + ".part"
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as e:
        print(f"[warn] {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return False
    os.replace(tmp, dest)
    return True


def _text_from(rec: dict, keys: list[str]) -> str:
    for k in keys:
        v = rec.get(k)
        if isinstance(v, str) and len(v.strip()) > 200:
            return v.strip()
    return ""


def normalize_parquet(parquet_path: str, out_path: str,
                      text_keys: list[str], max_docs: int | None = None,
                      join_fields: list[tuple[str, str]] | None = None) -> int:
    """Parquet rows -> SEP-separated clean-text docs."""
    pf = pq.ParquetFile(parquet_path)
    n = 0
    with open(out_path + ".tmp", "w", encoding="utf-8") as fh:
        for batch in pf.iter_batches(batch_size=512):
            cols = {name: batch.column(name).to_pylist()
                    for name in batch.schema.names}
            rows = range(len(cols[batch.schema.names[0]]))
            for i in rows:
                rec = {k: cols[k][i] for k in cols}
                text = ""
                if join_fields:                       # prompt/response pairs
                    parts = []
                    for a, b in join_fields:
                        va, vb = rec.get(a), rec.get(b)
                        if isinstance(va, str) and isinstance(vb, str) \
                                and len(vb.strip()) > 100:
                            parts.append(va.strip() + "\n" + vb.strip())
                    text = "\n\n".join(parts)
                else:
                    text = _text_from(rec, text_keys)
                text = _bc.clean_text(text)
                if len(text) < 400:                   # min useful doc size
                    continue
                fh.write(text[:200_000] + SEP)
                n += 1
                if max_docs and n >= max_docs:
                    break
            if max_docs and n >= max_docs:
                break
    os.replace(out_path + ".tmp", out_path)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=int, default=1,
                    help="parquet shards per source (1 = smallest viable)")
    ap.add_argument("--max-docs-per-source", type=int, default=120_000)
    ap.add_argument("--fineweb-bytes", type=int, default=400_000_000,
                    help="glue-language share from existing FineWeb-Edu txt")
    ap.add_argument("--out-dir", default="corpus_tech")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths, meta = {}, {}

    def add(name, url_tpl, norm_kwargs, shard_fmt="{i:02d}"):
        ok_all, parts = True, []
        for i in range(args.shards):
            fname = url_tpl.split("/")[-1]
            fname = fname.replace("00000", f"{i:05d}")
            dest = os.path.join("corpus", "raw_" + fname)
            url = url_tpl
            if "{i:02d}" == shard_fmt:
                url = url_tpl.format(i=i)
            if not _dl(url, dest):
                ok_all = False
                continue
            part_out = os.path.join(args.out_dir, f"{name}_{i}.txt")
            n = normalize_parquet(dest, part_out, **norm_kwargs)
            print(f"[ok  ] {name}[{i}]: {n:,} docs", flush=True)
            parts.append(part_out)
        if parts:
            merged = os.path.join(args.out_dir, f"{name}.txt")
            with open(merged, "wb") as out:
                for p_ in parts:
                    out.write(open(p_, "rb").read())
                    os.remove(p_)
            paths[name] = merged
            meta[name] = {"docs": sum(1 for _ in _bc.iter_docs(merged)),
                          "license": LICENSES[name]}
        return ok_all

    add("stackedu_py",
        SOURCES_URL["stackedu_py"],
        dict(text_keys=["content", "code", "text"], max_docs=args.max_docs_per_source))
    add("stackedu_md",
        SOURCES_URL["stackedu_md"],
        dict(text_keys=["content", "text", "markdown"], max_docs=args.max_docs_per_source))
    add("nemotron_code",
        SOURCES_URL["nemotron_code"],
        dict(join_fields=[("input", "output"), ("prompt", "response"),
                          ("question", "response")],
             max_docs=args.max_docs_per_source))
    add("nemotron_stem",
        SOURCES_URL["nemotron_stem"],
        dict(join_fields=[("input", "output"), ("prompt", "response"),
                          ("question", "response")],
             max_docs=args.max_docs_per_source))

    # glue language from already-normalized FineWeb-Edu
    fw_src = os.path.join(ROOT, "corpus", "fineweb-edu.txt")
    if os.path.exists(fw_src) and args.fineweb_bytes > 0:
        dst = os.path.join(args.out_dir, "fineweb_edu.txt")
        remaining = args.fineweb_bytes
        with open(fw_src, encoding="utf-8", buffering=1 << 22) as f, \
                open(dst + ".tmp", "w", encoding="utf-8") as o:
            while remaining > 0:
                block = f.read(1 << 23)
                if not block:
                    break
                block = block[:remaining]
                remaining -= len(block)
                o.write(block)
        os.replace(dst + ".tmp", dst)
        paths["fineweb_edu"] = dst
        meta["fineweb_edu"] = {
            "docs": sum(1 for _ in _bc.iter_docs(dst)),
            "license": LICENSES["fineweb_edu"]}

    src_json = {
        "sources": [{"name": k, "path": os.path.relpath(v, ROOT),
                     "license": meta[k]["license"]} for k in paths],
    }
    with open(os.path.join(args.out_dir, "sources.json"), "w") as f:
        json.dump(src_json, f, indent=2)
    print("\n[sources]", json.dumps(src_json, indent=2))
    print(f"\n[done] next: python scripts/build_corpus.py "
          f"--sources-json {args.out_dir}/sources.json "
          f"--tokenizer data_v3/tokenizer.json")


if __name__ == "__main__":
    main()
