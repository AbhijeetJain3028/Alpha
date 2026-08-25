#!/usr/bin/env python3
"""Indus training kernel - runs on Kaggle GPU notebooks.

Designed for EPHEMERAL environments: every ~SAVE_EVERY_STEPS the full
training state lands on the Hugging Face Hub, so any future session (this
kernel re-run, a fresh notebook, or this laptop) resumes exactly where it
left off. Nothing is lost when Kaggle recycles the machine.

Resume priority:
  1. newest ckpt-*.pt already mounted under /kaggle/input (prior outputs)
  2. newest local file in /kaggle/working
  3. ckpt-latest.pt downloaded from the HF Hub

Placeholders __HF_TOKEN__ / __PRESET__ etc. are replaced by scripts/kaggle_run.py
at push time; tokens are never committed to git.
"""

import glob
import json
import math
import os
import random
import signal
import sys
import time

import numpy as np
import torch

# ---------------------------------------------------------------- injected cfg
HF_TOKEN = "__HF_TOKEN__"
HF_REPO_ID = "__HF_REPO_ID__"
PRESET = "__PRESET__"
MAX_STEPS = int("__MAX_STEPS__")            # total optimizer steps across ALL runs
BATCH_SIZE = int("__BATCH_SIZE__")
TIME_BUDGET_MIN = float("__TIME_BUDGET_MIN__")   # stop safely before session kill
SAVE_EVERY_STEPS = int("__SAVE_EVERY_STEPS__")
EVAL_EVERY = int("__EVAL_EVERY__")
NUMBERED_EVERY = int("__NUMBERED_EVERY__")   # keep a permanent numbered ckpt
SEED = 1337

WORK = os.environ.get("INDUS_WORK", "/kaggle/working")
INPUT = os.environ.get("INDUS_INPUT", "/kaggle/input")

os.makedirs(WORK, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ------------------------------------------------------------- locate package+data
data_dir = None
for f in sorted(glob.glob(f"{INPUT}/**/train.bin", recursive=True)):
    data_dir = os.path.dirname(f)
    break
assert data_dir, ("no train.bin found under /kaggle/input - attach indus-data "
                  f"dataset. Mounted inputs: {os.listdir(INPUT) if os.path.isdir(INPUT) else 'none'}")
print("[setup] data dir:", data_dir)
sys.path.insert(0, data_dir)  # indus/ package ships inside the dataset

# dataset may ship the package as indus.zip - unpack before importing
if not os.path.isdir(os.path.join(data_dir, "indus")):
    zpath = os.path.join(data_dir, "indus.zip")
    if os.path.exists(zpath):
        import zipfile
        zipfile.ZipFile(zpath).extractall(data_dir)
        print("[setup] extracted", zpath)

from indus.config import get_config          # noqa: E402
from indus.data import TokenDataset, get_batch  # noqa: E402
from indus.model import IndusLM              # noqa: E402
from indus.tokenizer import BPETokenizer     # noqa: E402

print("python:", sys.version.split()[0], "| torch:", torch.__version__,
      "| cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0),
          "| cc:", torch.cuda.get_device_capability(0))

# ------------------------------------------------------------------------- hub api
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

hub = HfApi(token=HF_TOKEN or None)
INTERNET = bool(HF_TOKEN)


def hub_upload(path: str, path_in_repo: str, msg: str) -> bool:
    if not INTERNET:
        return False
    for attempt in range(3):
        try:
            hub.upload_file(path_or_fileobj=path, path_in_repo=path_in_repo,
                            repo_id=HF_REPO_ID, repo_type="model",
                            commit_message=msg)
            print(f"[hub ] uploaded {path_in_repo}")
            return True
        except Exception as e:
            print(f"[hub ] upload failed (attempt {attempt + 1}): {e}")
            time.sleep(10 * (attempt + 1))
    return False


try:
    hub.create_repo(HF_REPO_ID, repo_type="model", exist_ok=True,
                    private=True)
except Exception as e:
    print("[hub ] create_repo:", e)

# ---------------------------------------------------------------------- tokenizer
tok_path = os.path.join(data_dir, "tokenizer.json")
tok = BPETokenizer.load(tok_path)

# ------------------------------------------------------------------- resume logic
def candidate_ckpts() -> list[str]:
    pats = [f"{WORK}/ckpt-*.pt", f"{INPUT}/*/**/ckpt-*.pt", f"{INPUT}/*/ckpt-*.pt"]
    files: list[str] = []
    for pat in pats:
        files.extend(glob.glob(pat, recursive=True))
    if INTERNET:
        try:
            dl = hf_hub_download(HF_REPO_ID, "ckpt-latest.pt", repo_type="model",
                                 token=HF_TOKEN, force_download=False)
            files.append(dl)
        except Exception as e:
            print("[hub ] no remote checkpoint yet:", type(e).__name__)
    return files


def load_ckpt(path: str):
    print(f"[load] {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


best = None
best_step = -1
for f in candidate_ckpts():
    try:
        st = load_ckpt(f)
        if st.get("step", -1) > best_step:
            best, best_step = st, st.get("step", -1)
    except Exception as e:
        print(f"[load] skipping {f}: {e}")

start_step = 0
if best is not None:
    cfg = get_config(PRESET, **{k: v for k, v in best["config"].items()
                                if hasattr(get_config(PRESET), k)})
    model = IndusLM(cfg)
    model.load_state_dict(best["model"])
    start_step = best["step"]
    print(f"[load] resumed at step {start_step} "
          f"(val_loss={best.get('best_val', float('nan')):.4f})")
else:
    vocab_size = len(tok.vocab)
    cfg = get_config(PRESET, vocab_size=vocab_size)
    model = IndusLM(cfg)
    print("[init] fresh model")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4,
                              betas=(0.9, 0.95), eps=1e-8,
                              weight_decay=cfg.weight_decay)
if best is not None and best.get("optimizer"):
    optimizer.load_state_dict(best["optimizer"])
if best is not None and best.get("rng"):
    rng = best["rng"]
    torch.set_rng_state(rng["torch"].cpu())
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])
    if device == "cuda" and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
del best


def save_checkpoint(step: int, tag: str) -> None:
    """Atomic local save + durable upload to the Hub."""
    model.eval()
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg.to_dict(),
        "step": step,
        "best_val": globals().get("best_val", float("inf")),
        "rng": {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if device == "cuda" else None,
        },
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    model.train()
    tmp = os.path.join(WORK, f".{tag}.tmp")
    dst = os.path.join(WORK, tag)
    torch.save(payload, tmp)
    os.replace(tmp, dst)                      # atomic on same filesystem
    mb = os.path.getsize(dst) / 1e6
    print(f"[save] {dst} ({mb:.0f} MB, step {step})")
    hub_upload(dst, tag, f"Indus {cfg.name}: step {step}")


# ------------------------------------------------------------------ lr schedule
BASE_LR = 6e-4
WARMUP = 200
MIN_RATIO = 0.1


def lr_at(step: int) -> float:
    if step < WARMUP:
        return BASE_LR * (step + 1) / WARMUP
    if step >= MAX_STEPS:
        return BASE_LR * MIN_RATIO
    r = (step - WARMUP) / max(1, MAX_STEPS - WARMUP)
    return BASE_LR * MIN_RATIO + 0.5 * BASE_LR * (1 - MIN_RATIO) * (1 + math.cos(math.pi * r))


# mixed precision: fp16 on pre-Ampere GPUs (P100/T4), bf16 otherwise
cc_major = torch.cuda.get_device_capability(0)[0] if device == "cuda" else 0
AMP_DTYPE = torch.bfloat16 if cc_major >= 8 else torch.float16
USE_AMP = device == "cuda"
scaler = torch.amp.GradScaler(enabled=USE_AMP and AMP_DTYPE == torch.float16)
print(f"amp: {'off' if not USE_AMP else str(AMP_DTYPE).split('.')[-1]}")

train_ds = TokenDataset(os.path.join(data_dir, "train.bin"))
val_ds = TokenDataset(os.path.join(data_dir, "val.bin"))
tokens_per_step = BATCH_SIZE * cfg.block_size
print(f"model: {model.num_params() / 1e6:.2f}M params | "
      f"corpus: {train_ds.size:,} tok | {tokens_per_step:,} tok/step")


@torch.no_grad()
def run_eval(iters: int = 10) -> float:
    model.eval()
    losses = torch.zeros(iters)
    for i in range(iters):
        x, y = get_batch(val_ds, cfg.block_size, BATCH_SIZE, device)
        losses[i] = model(x, targets=y).loss.item()
    model.train()
    return losses.mean().item()


stop_now = False


def _graceful(signum, frame):
    global stop_now
    print(f"\n[signal] {signum} received - will save at next check")
    stop_now = True


signal.signal(signal.SIGTERM, _graceful)
signal.signal(signal.SIGINT, _graceful)

t0 = time.time()
deadline = t0 + TIME_BUDGET_MIN * 60
best_val = float(globals().get("best_val", float("inf")))
log_every = 25

model.train()
final_step = start_step
for step in range(start_step, MAX_STEPS):
    final_step = step + 1
    now = time.time()

    # ---- durability & time checks BEFORE burning more compute
    if now > deadline:
        print(f"[time] budget of {TIME_BUDGET_MIN:.0f}min reached - saving & exiting")
        break
    if stop_now:
        print("[signal] stopping early")
        break

    lr = lr_at(step)
    for g in optimizer.param_groups:
        g["lr"] = lr

    x, y = get_batch(train_ds, cfg.block_size, BATCH_SIZE, device)
    with torch.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=USE_AMP):
        out = model(x, targets=y)
    loss = out.loss

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()

    done = step + 1 - start_step
    rate = done * tokens_per_step / max(time.time() - t0, 1e-6)
    if step % log_every == 0 or step == MAX_STEPS - 1:
        eta_min = (MAX_STEPS - step - 1) * max(time.time() - t0, 1e-6) / max(done, 1) / 60
        print(f"iter {step:5d}/{MAX_STEPS} | loss {loss.item():.4f} | "
              f"ppl {math.exp(min(loss.item(), 20)):7.2f} | lr {lr:.2e} | "
              f"{rate / 1e3:6.1f}k tok/s | eta {eta_min:5.0f}m | "
              f"left {(deadline - time.time()) / 60:5.0f}m")

    if step > 0 and step % EVAL_EVERY == 0:
        vl = run_eval()
        best_val = min(best_val, vl)
        print(f"[eval ] iter {step}: val_loss {vl:.4f} (best {best_val:.4f})")

    if (step + 1) % SAVE_EVERY_STEPS == 0 or step == MAX_STEPS - 1:
        save_checkpoint(step + 1, "ckpt-latest.pt")
    if (step + 1) % NUMBERED_EVERY == 0:
        save_checkpoint(step + 1, f"ckpt-{step + 1:06d}.pt")

# ------------------------------------------------------------------------ finish
if final_step == start_step:
    print("[warn] no optimizer steps were run this session")
vl = run_eval()
best_val = min(best_val, vl)
print(f"[final] val_loss {vl:.4f} after step {final_step}")

save_checkpoint(final_step, "ckpt-final.pt")
save_checkpoint(final_step, "ckpt-latest.pt")

# keep only recent numbered checkpoints in working dir (output size hygiene)
for old in sorted(glob.glob(f"{WORK}/ckpt-*.pt"))[:-4]:
    try:
        os.remove(old)
    except OSError:
        pass

# generate samples so logs show qualitative progress
gen_ids = tok.encode("Once upon a time")
xg = torch.tensor([gen_ids], dtype=torch.long, device=device)
yg = model.generate(xg, max_new_tokens=300, temperature=0.8, top_k=50,
                    endoftext_id=tok.special_tokens.get("<|endoftext|>"))
sample = tok.decode(yg[0].tolist())
print("\n===== sample =====\n" + sample + "\n==================")
with open(os.path.join(WORK, "samples.txt"), "w") as f:
    f.write(f"step {final_step}\n\n{sample}\n")
hub_upload(os.path.join(WORK, "samples.txt"), f"samples/step-{final_step}.txt",
           f"Samples at step {final_step}")

total_h = (time.time() - t0) / 3600
print(f"done in {total_h:.2f}h | state is safe on the Hub ({HF_REPO_ID})")
