#!/usr/bin/env python3
"""Kaggle orchestrator for training Indus on ephemeral free GPUs.

Durability model (nothing is ever lost when Kaggle recycles a notebook):

    Kaggle GPU kernel  --(every N steps)-->  Hugging Face Hub repo
          ^                                            |
          |  resume ckpt-latest.pt                     v
          +------------------------------------ durable memory

Commands:
  dataset        create/update the indus-data dataset (corpus + code)
  push           render + submit the training kernel (starts a run)
  status         show current kernel status
  watch          poll until the run finishes, then print tail of log
  output         download the latest run's output files

Credentials (never committed): env vars KAGGLE_API_TOKEN and HF_TOKEN,
or a sourced env file passed via --env.

Typical loop:
  python scripts/kaggle_run.py dataset
  python scripts/kaggle_run.py push
  python scripts/kaggle_run.py watch
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "kaggle" / "build"
KERNEL_TEMPLATE = ROOT / "scripts" / "kernel_indus_train.py"

KAGGLE_USER = "abhijeetjain3027"
DATA_SLUG = "indus-data"
KERNEL_SLUG = "indus-train"


def load_api():
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    return api


# --------------------------------------------------------------------------- dataset
def cmd_dataset(args) -> None:
    api = load_api()
    data_dir = Path(args.data_dir or (ROOT / "data"))
    for req in ("train.bin", "val.bin", "tokenizer.json"):
        if not (data_dir / req).exists():
            raise SystemExit(f"missing {data_dir / req} - run scripts/prepare_data.py first")

    folder = BUILD / DATA_SLUG
    if folder.exists():
        shutil.rmtree(folder)
    (folder / "indus").mkdir(parents=True)

    # corpus + tokenizer
    for name in ("train.bin", "val.bin", "tokenizer.json"):
        shutil.copy2(data_dir / name, folder / name)
    # the model package itself ships with the data
    for f in (ROOT / "indus").glob("*.py"):
        shutil.copy2(f, folder / "indus" / f.name)
    # SFT trainer module used by the SFT kernel
    shutil.copy2(ROOT / "scripts" / "train_sft.py", folder / "train_sft.py")
    meta = {
        "title": "Indus LLM training data",
        "id": f"{KAGGLE_USER}/{DATA_SLUG}",
        "licenses": [{"name": "CC0-1.0"}],
    }
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))

    existing = [d for d in (api.dataset_list(mine=True) or [])
                if getattr(d, "ref", "") == f"{KAGGLE_USER}/{DATA_SLUG}"]
    if existing:
        # NOTE: create_new on an existing slug silently no-ops - must version
        api.dataset_create_version(folder=str(folder),
                                   version_notes="refresh data/code",
                                   dir_mode="zip")
        print(f"[kaggle] updated dataset {KAGGLE_USER}/{DATA_SLUG}")
    else:
        # dir_mode="zip": the indus/ package ships as indus.zip (kernel unzips)
        api.dataset_create_new(folder=str(folder), dir_mode="zip")
        print(f"[kaggle] created dataset {KAGGLE_USER}/{DATA_SLUG}")


# ------------------------------------------------------------------------ kernel
def render_kernel(args) -> Path:
    if not args.hf_token or not args.hf_repo:
        raise SystemExit("--hf-token/--hf-repo required (or set HF_TOKEN/HF_REPO)")
    folder = BUILD / KERNEL_SLUG
    folder.mkdir(parents=True, exist_ok=True)

    src = KERNEL_TEMPLATE.read_text()
    src = src.replace("__HF_TOKEN__", args.hf_token)
    src = src.replace("__HF_REPO_ID__", args.hf_repo)
    src = src.replace("__PRESET__", args.preset)
    src = src.replace("__MAX_STEPS__", str(args.max_steps))
    src = src.replace("__BATCH_SIZE__", str(args.batch_size))
    src = src.replace("__TIME_BUDGET_MIN__", str(args.time_budget_min))
    src = src.replace("__SAVE_EVERY_STEPS__", str(args.save_every_steps))
    src = src.replace("__EVAL_EVERY__", str(args.eval_every))
    src = src.replace("__NUMBERED_EVERY__", str(args.numbered_every))

    script = folder / "indus_train_kernel.py"
    script.write_text(src)
    os.chmod(script, 0o600)  # contains the HF token

    meta = {
        "id": f"{KAGGLE_USER}/{KERNEL_SLUG}",
        "title": "indus-train",
        "code_file": "indus_train_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_internet": "true" if not args.no_internet else "false",
        "machine_shape": args.accelerator,
        "dataset_sources": [f"{KAGGLE_USER}/{DATA_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return folder


def cmd_push(args) -> None:
    api = load_api()
    folder = render_kernel(args)
    res = api.kernels_push(folder=str(folder), acc=args.accelerator)
    print("[kaggle] pushed:", getattr(res, "url", res))
    print(f"monitor: https://www.kaggle.com/{KAGGLE_USER}/{KERNEL_SLUG}")


def cmd_status(_args) -> None:
    api = load_api()
    st = api.kernels_status(f"{KAGGLE_USER}/{KERNEL_SLUG}")
    print(json.dumps(st, indent=2, default=str))


def cmd_watch(args) -> None:
    api = load_api()
    terminal = {"COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"}
    while True:
        st = api.kernels_status(f"{KAGGLE_USER}/{KERNEL_SLUG}")
        status = st.get("status", "?") if isinstance(st, dict) else str(st)
        print(time.strftime("[%H:%M:%S]"), status, flush=True)
        if str(status).upper() in terminal:
            break
        time.sleep(max(30, args.interval))
    cmd_output(args)


def cmd_output(args) -> None:
    api = load_api()
    out = Path(args.out_dir or (ROOT / "kaggle" / "output"))
    out.mkdir(parents=True, exist_ok=True)
    files, token = api.kernels_output(f"{KAGGLE_USER}/{KERNEL_SLUG}",
                                      path=str(out), quiet=False)
    print("downloaded:", files)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dataset")
    p.add_argument("--data-dir", default=None)
    p.set_defaults(fn=cmd_dataset)

    p = sub.add_parser("push")
    p.add_argument("--preset", default="indus-tiny")
    p.add_argument("--max-steps", type=int, default=7000,
                   help="total steps across all sessions (~220M tokens, chinchilla-ish)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--time-budget-min", type=float, default=430.0,
                   help="stop saving-and-exiting before Kaggle's hard kill")
    p.add_argument("--save-every-steps", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--numbered-every", type=int, default=2000)
    p.add_argument("--no-internet", action="store_true")
    p.add_argument("--accelerator", default="NvidiaTeslaT4",
                   choices=["NvidiaTeslaT4", "NvidiaTeslaP100"])
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="defaults to $HF_TOKEN")
    p.add_argument("--hf-repo", default=os.environ.get("HF_REPO",
                   "AbhijeetJain4075/indus-llm"))
    p.set_defaults(fn=cmd_push)

    p = sub.add_parser("status")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("watch")
    p.add_argument("--interval", type=float, default=60)
    p.add_argument("--out-dir", default=None)
    p.set_defaults(fn=cmd_watch)

    p = sub.add_parser("output")
    p.add_argument("--out-dir", default=None)
    p.set_defaults(fn=cmd_output)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
