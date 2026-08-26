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

from indus.autonomous import clean_text            # noqa: E402

HF = "https://huggingface.co/datasets"
SOURCES_URL = {
    "openwebmath": HF + "/open-web-math/open-web-math/resolve/main/"
                   "data/train-{shard}-of-00114-5a023365406cb9c4.parquet",
}
NEMOTRON_V1 = {
    "nemotron_code": "SFT/code/code_v1.1.jsonl",
    "nemotron_math": "SFT/math/math_v1.1.jsonl",
    "nemotron_sci": "SFT/science/science.jsonl",
}
LICENSES = {
    "stackedu_py": "Stack-v2 terms (permissive-only slice)",
    "stackedu_md": "Stack-v2 terms (permissive-only slice)",
    "openwebmath": "ODC-BY-1.0",
    "nemotron_code": "CC-BY-4.0",
    "nemotron_math": "CC-BY-4.0",
    "nemotron_sci": "CC-BY-4.0",
    "fineweb_edu": "ODC-BY-1.0",
}
PERMISSIVE = {"mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause",
              "isc", "0bsd", "unlicense", "wtfpl"}
SWH_RAW = "https://archive.softwareheritage.org/api/1/blob/{id}/raw/"


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
                text = clean_text(text)
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



def swh_fetch_stackedu(parquet_path: str, out_path: str, name: str,
                       limit: int) -> int:
    """Metadata rows -> fetch real file contents from Software Heritage,
    keeping only permissive-license, high-quality files."""
    from concurrent.futures import ThreadPoolExecutor
    pf = pq.ParquetFile(parquet_path)
    print(f"[swh ] scanning {pf.metadata.num_rows:,} rows (streaming)...",
          flush=True)
    rows = []
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=[
            "blob_id", "path", "int_score", "license_type"])
        d = {n_: tbl.column(n_).to_pylist()
             for n_ in tbl.schema.names}
        for b, pth, s, l_ in zip(d["blob_id"], d["path"],
                                 d["int_score"], d["license_type"]):
            if l_.lower() == "permissive" and s >= 4 \
                    and pth.lower().endswith(
                        (".py",) if name.endswith("_py")
                        else (".md", ".rst")):
                rows.append((b, pth, s))
        del tbl
        print(f"[swh ]   rowgroup {rg + 1}/{pf.num_row_groups}: "
              f"{len(rows)} candidates so far", flush=True)
    rows.sort(key=lambda r: -r[2])
    rows = rows[:limit]
    ext_ok = (".py",) if name.endswith("_py") else         (".md", ".rst", ".txt")
    rows = [r for r in rows if r[1].lower().endswith(ext_ok)]
    print(f"[swh ] {name}: {len(rows)} candidates "
          f"(permissive, score>=4)", flush=True)

    def grab(row):
        bid, pth = row[0], row[1]
        try:
            req = urllib.request.Request(SWH_RAW.format(id=bid),
                                         headers={"User-Agent": "indus-tech"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
            text = data.decode("utf-8", errors="replace")
            return (pth, text) if len(text) > 400 else None
        except Exception:
            return None

    n = 0
    seen = set()
    ledger = out_path + ".ledger"
    if os.path.exists(ledger):
        seen = set(open(ledger).read().split())
    mode = "a" if seen else "w"
    fresh = 0
    with open(out_path, mode, encoding="utf-8") as fh, \
            open(ledger, "a", encoding="utf-8") as led:
        with ThreadPoolExecutor(max_workers=16) as ex:
            for res in ex.map(grab, rows):
                fresh += 1
                if not res:
                    continue
                key = hash(res[1][:2000])
                if key in seen:
                    continue
                seen.add(key)
                fh.write(f"# {res[0]}\n"
                         + clean_text(res[1])[:100_000] + SEP)
                led.write(res[0] + "\n")
                n += 1
                if n % 250 == 0:
                    fh.flush(); led.flush()
                    print(f"[swh ] kept {n} ({fresh} fetched)",
                          flush=True)
    return n



def normalize_jsonl(path: str, out_path: str, limit: int) -> int:
    """Nemotron v1 style: records with input/output or prompt/response."""
    n = 0
    seen = set()
    with open(path, encoding="utf-8") as f, \
            open(out_path + ".tmp", "w", encoding="utf-8") as o:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            a = (r.get("input") or r.get("prompt") or
                 r.get("question") or "").strip()
            b = (r.get("output") or r.get("response") or
                 r.get("answer") or "").strip()
            if len(b) < 120:
                continue
            doc = (a + "\n" + b).strip() if a else b
            key = hash(doc[:1500])
            if key in seen:
                continue
            seen.add(key)
            o.write(clean_text(doc)[:120_000] + SEP)
            n += 1
            if n >= limit:
                break
    os.replace(out_path + ".tmp", out_path)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", type=int, default=1,
                    help="parquet shards per source (1 = smallest viable)")
    ap.add_argument("--max-docs-per-source", type=int, default=120_000)
    ap.add_argument("--swh-limit", type=int, default=15_000,
                help="max permissive GitHub files fetched from SWH per source")
    ap.add_argument("--fineweb-bytes", type=int, default=400_000_000,
                    help="glue-language share from existing FineWeb-Edu txt")
    ap.add_argument("--out-dir", default="corpus_tech")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    paths, meta = {}, {}
    lim = args.max_docs_per_source

    # 1) Stack-Edu via Software Heritage (GitHub-derived)
    se_meta = "corpus/raw_train-00000-of-00005.parquet"   # python metadata
    md_meta = "corpus/raw_train-00000-of-00005-md.parquet"
    if _dl("https://huggingface.co/datasets/HuggingFaceTB/stack-edu/resolve/main/Markdown/train-00000-of-00005.parquet", md_meta):
        for name, metaf, ext in (("stackedu_py", se_meta, (".py",)),
                                 ("stackedu_md", md_meta, (".md", ".rst"))):
            if not os.path.exists(metaf):
                continue
            outp = os.path.join(args.out_dir, name + ".txt")
            n = swh_fetch_stackedu(metaf, outp, name,
                                   limit=args.swh_limit)
            if n:
                paths[name] = outp
                meta[name] = {"docs": n, "license": LICENSES[name]}
                print(f"[ok  ] {name}: {n:,} docs", flush=True)

    # 2) OpenWebMath sample shard
    owm = os.path.join("corpus", "raw_owm_0.parquet")
    if _dl(SOURCES_URL["openwebmath"].format(shard="00000"), owm):
        outp = os.path.join(args.out_dir, "openwebmath.txt")
        n = normalize_parquet(owm, outp, text_keys=["text"],
                              max_docs=lim)
        if n:
            paths["openwebmath"] = outp
            meta["openwebmath"] = {"docs": n, "license": LICENSES["openwebmath"]}
            print(f"[ok  ] openwebmath: {n:,} docs", flush=True)

    # 3) Nemotron v1 JSONLs (CC-BY-4.0)
    for key, rel in NEMOTRON_V1.items():
        dest = os.path.join("corpus", "raw_" + rel.replace("/", "_"))
        if _dl(HF + "/nvidia/Llama-Nemotron-Post-Training-Dataset/resolve/main/" + rel, dest):
            outp = os.path.join(args.out_dir, key + ".txt")
            n = normalize_jsonl(dest, outp, limit=lim)
            if n:
                paths[key] = outp
                meta[key] = {"docs": n, "license": LICENSES[key]}
                print(f"[ok  ] {key}: {n:,} docs", flush=True)

    # 4) FineWeb-Edu glue
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
