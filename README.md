# Indus

A general-purpose large language model built **from scratch** — no pretrained
weights, no tokenizer libraries, no training frameworks. Just PyTorch.

## Architecture

Indus is a modern decoder-only transformer:

| Component | Choice |
|---|---|
| Position encoding | Rotary embeddings (RoPE) |
| Normalization | RMSNorm, pre-norm |
| Attention | Causal multi-head with Grouped-Query Attention (GQA) |
| Feed-forward | SwiGLU |
| Biases | None (Linear layers without bias) |
| Embeddings | Tied input/output weights |

## Presets

| Preset | Params | Layers | Width | Context | Trains on |
|---|---|---|---|---|---|
| `indus-nano` | ~1.3M | 4 | 128 | 256 | laptop CPU (minutes) |
| `indus-tiny` | ~11M | 6 | 384 | 512 | CPU overnight / GPU minutes |
| `indus-small` | ~42M | 12 | 576 | 1024 | small GPU |
| `indus-medium` | ~150M | 16 | 960 | 2048 | GPU |

All hyperparameters can be overridden from the CLI on top of any preset.

## Quickstart

```bash
pip install -r requirements.txt

# 1. get a corpus + train the BPE tokenizer + encode to binary tokens
python scripts/prepare_data.py --out data          # streams ~25MB of TinyStories
#    ...or use your own text:
python scripts/prepare_data.py --out data --input "mycorpus/*.txt"

# 2. train
python scripts/train.py --data-dir data --preset indus-nano --max-iters 2000
python scripts/train.py --data-dir data --preset indus-tiny --max-iters 10000

# 3. generate
python scripts/generate.py --ckpt checkpoints/ckpt.pt --prompt "Once upon a time"
```

Interactive chat mode:

```bash
python scripts/generate.py --ckpt checkpoints/ckpt.pt --chat
```

## Project layout

```
indus/
  config.py       IndusConfig dataclass + presets
  tokenizer.py    byte-level BPE trained from scratch
  model.py        the transformer (RoPE, GQA, SwiGLU, RMSNorm)
  data.py         memmap token batching
  demo_corpus.py  tiny offline fallback corpus
scripts/
  prepare_data.py corpus download + tokenization pipeline
  train.py        AdamW + warmup/cosine LR, AMP on CUDA, checkpointing
  generate.py     sampling (temperature/top-k) and chat loop
tests/
  test_indus.py   smoke tests (run: python tests/test_indus.py)
```

## Training on Kaggle free GPUs (ephemeral-proof)

Kaggle notebooks are temporary, so Indus treats the **Hugging Face Hub as
durable memory**: the training kernel uploads model+optimizer+RNG state every
500 steps, and any future session resumes from `ckpt-latest.pt`. Nothing is
lost when Kaggle recycles the machine.

```
Kaggle GPU (T4)  --ckpt every 500 steps-->  HF repo AbhijeetJain4075/indus-llm
      ^                                              |
      +-------- resume ckpt-latest.pt ---------------+
```

One-time setup:

```bash
export KAGGLE_API_TOKEN=<your KGAT token>     # kaggle.com/settings/api
export HF_TOKEN=<your hf token>               # huggingface.co/settings/tokens
python scripts/prepare_data.py --out data     # corpus + tokenizer (done once)
```

Train / resume / repeat:

```bash
python scripts/kaggle_run.py dataset          # ship data+code to Kaggle
python scripts/kaggle_run.py push             # launch T4 run (auto-resumes)
python scripts/kaggle_run.py watch            # poll until finished, pull logs
python scripts/kaggle_run.py push             # ...repeat until target steps
python scripts/generate.py --ckpt <downloaded ckpt> --prompt "Once upon a time"
```

Defaults: preset `indus-tiny` (~11M params), batch 64 × ctx 512 = 32k tok/step,
7,000 total steps ≈ 229M tokens (Chinchilla-optimal ~20 tok/param), fp16 on T4,
430-min safe time budget per session, checkpoints pushed to a **private** HF
repo every 500 steps + permanent numbered snapshots every 2,000 steps.

The research grounding every design decision lives in [docs/PAPERS.md](docs/PAPERS.md).

## Pipeline v2 (real scale)

`scripts/build_corpus.py` assembles a multi-source, license-checked pretraining
mix (see `corpus/manifest.json` for counts + licenses), trains a 16k BPE, and
encodes ~1B tokens into `data_v2/`:

```bash
python scripts/build_corpus.py            # resumable; caches per source
```

| Source | License | Docs | Role |
|---|---|---|---|
| TinyStories | CDLA-Sharing-1.0 | ~545k | narrative fluency |
| FineWeb-Edu sample | ODC-BY-1.0 | ~726k | educational web |
| WikiText-103 | CC-BY-SA-3.0 | ~260k | factual prose |

Then train `indus-base` (~38M params, GQA, ctx 1024) on Kaggle T4s:

```bash
export KAGGLE_API_TOKEN=... HF_TOKEN=...
python scripts/kaggle_run.py dataset --data-dir data_v2
python scripts/kaggle_run.py push --preset indus-base --batch-size 16 \
    --max-steps 48000          # ~800M tokens ≈ Chinchilla-optimal
python scripts/kaggle_run.py watch
```

### Chat stage (SFT)

```bash
python scripts/build_sft_data.py              # OASST1 + Dolly + SmolTalk
python scripts/train_sft.py                   # pulls base ckpt from the Hub
python scripts/generate.py --which ckpt-sft.pt --chat
```

### Evaluation

```bash
python scripts/evaluate.py                    # WikiText PPL + LAMBADA + ARC-Easy
```

The research grounding every design decision lives in [docs/PAPERS.md](docs/PAPERS.md).

## Autonomy — research the web, learn, answer with citations

Indus can autonomously research a topic, remember it in a retrieval memory,
fine-tune on what it found (eval-gated with replay-buffer rollback so it
never forgets its base skills), and answer grounded questions with citations:

```bash
# one-shot cycles
python scripts/auto_learn.py --topics "Eiffel Tower,Photosynthesis"

# interactive
python scripts/auto_learn.py
```

In the web UI, hit `POST /api/research {"session_id": "...", "topic": "..."}`
(SSE streams sources → learning stats → grounded answer), or send any chat
message with `"research": true` to answer from the knowledge store.
Design doctrine lives in [docs/PLAYBOOK.md](docs/PLAYBOOK.md):
knowledge lives outside the network (FTS5 BM25 retriever); the tiny model
shapes queries and composes grounded answers; self-training rolls back
unless held-out perplexity improves.

## Indus-Kernel integration

Selected, peak-value ports from the [Indus-Kernel](https://github.com/Abhijeetjain4075/Indus-Kernel)
control plane (the full 40-subsystem stack can drive Indus through the bridge):

| Port | File | Purpose |
|---|---|---|
| Auditable research contract | `indus/research_contract.py` | provenance-tied claims; refuses without evidence |
| Sparse MoE-SwiGLU | `indus/moe.py` + `config.use_moe` | dormant expert routing for the next pretrain |
| Constitutional scaffold | `indus/constitution.py` | principles prefix + deterministic PII/harm lint + regenerate |
| stdio tool server (A2A/MCP-style) | `scripts/indus_mcp_server.py` | lets any agent/kernel call `chat`/`research`/`answer`/`info` |

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python scripts/indus_mcp_server.py
```

## GPU fine-tuning (chat) on Kaggle

```bash
export KAGGLE_API_TOKEN=... HF_TOKEN=...
python scripts/kaggle_run.py dataset      # ships corpus + SFT corpus + code
python scripts/kaggle_run.py push-sft     # T4 run -> ckpt-sft.pt to the Hub
python scripts/kaggle_run.py watch --kernel indus-sft
python scripts/kaggle_run.py status --kernel indus-sft
```

The training kernel early-stops on validation patience and always keeps the
val-best weights (`ckpt-best.pt`); uploads are content-hash deduped; the HF
token resolves env-var → Kaggle Secret → placeholder (never required in
kernel source). Full doctrine: [docs/PLAYBOOK.md](docs/PLAYBOOK.md),
paper-by-paper grounding: [docs/PAPERS.md](docs/PAPERS.md).

## Notes

- The tokenizer is intentionally simple/pure-Python; for multi-GB corpora swap
  in a Rust BPE implementation — the model doesn't care.
- Training uses `torch.autocast` (bfloat16) automatically when a CUDA device
  is present; everything falls back cleanly to fp32/CPU.
- Checkpoints store both weights and the full config, so generation only needs
  the `.pt` file plus the tokenizer JSON.
