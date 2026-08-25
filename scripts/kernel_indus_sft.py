#!/usr/bin/env python3
"""Indus SFT (chat fine-tuning) kernel - runs on Kaggle GPU.

Loads the latest pretrained ckpt-latest.pt (from input mounts or the Hub),
fine-tunes on the masked SFT corpus shipped in the indus-data dataset, and
pushes ckpt-sft.pt back to the Hub. Short job (<1h) - no resume machinery
beyond pulling the newest base checkpoint.
"""

import glob
import os
import sys

import torch

INPUT = os.environ.get("INDUS_INPUT", "/kaggle/input")
WORK = os.environ.get("INDUS_WORK", "/kaggle/working")
os.makedirs(WORK, exist_ok=True)

HF_TOKEN = "__HF_TOKEN__"
HF_REPO_ID = "__HF_REPO_ID__"
STEPS = int("__SFT_STEPS__")
BATCH_SIZE = int("__SFT_BATCH__")

data_dir = None
for f in sorted(glob.glob(f"{INPUT}/**/train.bin", recursive=True)):
    data_dir = os.path.dirname(f)
    break
assert data_dir, "indus-data dataset not mounted"
sys.path.insert(0, data_dir)
if not os.path.isdir(os.path.join(data_dir, "indus")):
    import zipfile
    z = os.path.join(data_dir, "indus.zip")
    if os.path.exists(z):
        zipfile.ZipFile(z).extractall(data_dir)

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

hub = HfApi(token=HF_TOKEN or None)

# pick freshest base: local mounts first, else hub
base = None
cands = sorted(glob.glob(f"{INPUT}/**/ckpt-latest.pt", recursive=True),
               key=os.path.getmtime)
if cands:
    base = cands[-1]
else:
    base = hf_hub_download(HF_REPO_ID, "ckpt-latest.pt", repo_type="model",
                           token=HF_TOKEN)
print("[load] base:", base)

sys.argv = ["train_sft.py",
            "--base-ckpt", base,
            "--tokenizer", os.path.join(data_dir, "tokenizer.json"),
            "--data-dir", data_dir,
            "--out", os.path.join(WORK, "ckpt-sft.pt"),
            "--hf-repo", HF_REPO_ID,
            "--steps", str(STEPS),
            "--batch-size", str(BATCH_SIZE)]

train_sft.main()
