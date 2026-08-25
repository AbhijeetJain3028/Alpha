# Indus Peak Playbook — Free-Compute Training, Serving & Autonomy

Distilled from the August 2026 deep-research sweep (Kaggle quota mechanics,
Hugging Face infrastructure, OSS acceleration stack, capability-per-FLOP
papers, license-clean data catalog, and autonomous-web-learning lineage).
Every claim below was verified against primary sources; key links at the end.

---

## 1. Kaggle free tier — exact mechanics

| Fact | Value | Consequence |
|---|---|---|
| GPU quota | **30 h/week** (phone-verified), resets weekly | Budget runs like compute contracts |
| Session caps | 9 h GPU / 12 h CPU; background "Save & Run All" | Our 430-min self-budget is correct |
| Hardware | **GPU T4 x2** is the only valid choice | P100 (sm_60) is refused by modern PyTorch wheels |
| Persistence | `/kaggle/working` 20 GB output versioning | Belt-and-suspenders behind HF Hub |
| RAM | ~13 GB usable host | Keep dataloader workers ≤2 |

**Durability doctrine (implemented in `scripts/kernel_indus_train.py`):**
HF Hub = durable memory. Resume priority inputs→working→Hub; atomic saves;
full optimizer+RNG state; **early-stopping with patience** now uploads
`ckpt-best.pt` at every val improvement so the overfit-tail problem
(v1's val U-curve: best 2.306 @1.5k, final 3.68 @7k) can never ship again.
Uploads are content-hash deduped (no more duplicate commits).
The HF token resolves env-var → Kaggle Secret → placeholder, so it no longer
needs to live in kernel source.

## 2. T4 (cc 7.5) acceleration truth table

| Do | Don't |
|---|---|
| **fp16 AMP + GradScaler** | bf16 anywhere — `is_bf16_supported()` lies on Turing; measured **4.3× slower** |
| SDPA mem-efficient backend (`is_causal=True`) — GQA supported via `enable_gqa=True` | `pip install flash-attn` (sm80+ only), FA3 (Hopper-only), FlexAttention (no win ≤2k ctx) |
| `AdamW(fused=True)` — tiny models are launch-bound | bitsandbytes/Adafactor — memory isn't binding ≤150M params |
| `torch.compile`: ~1.2× train, up to ~2× decode w/ `reduce-overhead` (static shapes!) | DeepSpeed/FSDP/vLLM/TGI — wrong problem class, vLLM V1 needs sm80+ |
| Liger kernels selectively, fp16-tested on T4 only | GGUF quantization below Q8_0 at ≤50M params — buys nothing, hurts tiny models most |

## 3. Hugging Face Hub — free-tier numbers that matter

- Private storage: **100 GB free** (PRO 1 TB). Xet backend (July 2025+):
  chunk-level global dedup → re-pushing a checkpoint uploads only deltas.
- Rate limits per 5-min window (free auth'd): 1,000 API / 5,000 file GETs.
  **Always set HF_TOKEN on Kaggle** — anonymous buckets are shared per IP.
- Commit etiquette: batch pushes every 30–60 min, keep ≤3 revisions,
  `super_squash_history()` at milestones (history degrades after few k commits).
- Serving: Spaces free CPU Basic = 2 vCPU / 16 GB, sleeps after 48h idle.
  ZeroGPU ≈ minutes/day — inference demos only. Jobs = paid ($0.01/hr floor).

**Serving path (peak):** emit Llama-style tensor names → `convert_hf_to_gguf.py`
→ F16 → Q8_0 → `llama-server` in a Docker Space (OpenAI-compatible HTTP).
Expected: 40M model ≈ 60–150 tok/s on free 2-vCPU; 10M model faster still.
Fallback: ONNX Runtime int8.

## 4. Capability-per-FLOP recipe (papers → actions)

- **Vocab tax first**: byte-BPE 8k (11M) / 12k (40M); untied lm_head forbidden <300M.
- **Deep-narrow wins small** (MobileLLM): indus-11M d=288/L=14; indus-40M d=512/L=16.
- **Muon optimizer** (orthogonalized momentum, Kimi-K2-scale validated) on ≥2D
  matrices + AdamW on embeddings/norms; WSD schedule with **high-quality cooldown**
  final 5–10% of tokens (MiniCPM).
- Over-train small models (SmolLM2 doctrine): inference is nearly free; cycle a
  ~500M-token unique pool ≤4 epochs instead of 42 epochs over 5M tokens.
- Multi-token prediction (k=2) heads; checkpoint averaging + model soups at end.
- Distillation ladder (all Apache/MIT teachers): data-distill from
  Qwen2.5-0.5B/Qwen3-0.6B or SmolLM2-360M first; logit-KD (GKD/DistiLLM skew-KL)
  only if we adopt the teacher's tokenizer. R1 lesson: distilled students beat RL peers.

### Pretrain mix (unique-token shares)
45% FineWeb-Edu · 18% Cosmopedia-v2 stories/textbooks · 12% SimpleWiki+enwiki-leads ·
8% FineMath4+/OpenWebMath · 8% Stack-Edu py/md · 6% Nemotron-CC Diverse-QA ·
3% smoltalk2 tool-trace snippets.

### SFT mix (license-clean core)
smol-smoltalk (core) + smoltalk2 tool slices + Nemotron-PT-v2 chat/stem subsample
(CC-BY-4.0) + UltraFeedback-binarized DPO finish. Avoid: OpenHermes (no license),
Magpie sets (CC-BY-NC), No Robots (NC), The Pile (Books3 taint), C4 (dominated).

## 5. Autonomy architecture (the honest one)

Research lineage: REALM/RAG/RA-DIT (knowledge external), Toolformer/Gorilla
(tool calls), STaR/Quiet-STaR (self-improvement), Self-RAG/Search-R1
(search-native LM), WebGPT (browser RL), FreshLLMs (factuality), EWC/replay
(forgetting). What an 11M model actually does — verified by our own tests:

| Job | Owner | Implementation |
|---|---|---|
| Carry facts | **retriever** | SQLite FTS5 BM25 store (`KnowledgeStore`) |
| Query shaping, extraction, composition | **Indus** | `grounded_prompt` + generation |
| Learn without forgetting | eval-gate + replay | `SelfLearner`: rollback unless held-out probe improves; 35% replay from base corpus |
| Search protocol | scaffold, not RL | ReAct-shaped loop; deterministic grammar; banned: outcome-RL search policies at this scale |

Live code paths: `indus/autonomous.py`, `scripts/auto_learn.py`,
webapp `/api/research`. Deploy gate each cycle: probe Δ>0, store growth,
citation presence, TinyStories coherence spot-check.

## 6. Source index

Kaggle GPU usage: kaggle.com/docs/efficient-gpu-usage · bf16-on-T4 trap:
earino.github.io (applied-deep-learning) · SDPA backends: PyTorch blog
"Accelerated PyTorch 2" · Liger: arXiv:2410.10989 (+ issue #51 cc7.5 caveat) ·
torch.compile/CUDA-graphs: PyTorch tutorial · GGUF converter:
ggml.org/llama.cpp convert_hf_to_gguf.py · CPU tok/s anchor:
samarkanov.info (Feb 2026) · Hub storage/rate-limits/Xet/Jobs/ZeroGPU:
huggingface.co/docs · Trackio: huggingface.co/docs/trackio · Papers:
Kaplan'20 (2001.08361), Chinchilla (2203.15556), Muennighoff'23 (2305.16264),
phi-1 (2306.11644), TinyStories (2305.07759), SmolLM2 (2502.02737), MobileLLM
(2402.14905), Muon (Keller Jordan'24 + Kimi K2 2507.20534), MiniCPM/WSD
(2404.06395), MTP (2404.19737), Model Soups (2203.05482), GQA (2305.13245),
REALM (2002.08909), RA-DIT (2212.10517), Self-RAG (2310.11511), Toolformer
(2302.04761), STaR (2203.14465), Quiet-STaR (2403.09629), Search-R1
(2503.09516), WebGPT (2112.09332), FreshLLMs (2310.03214), EWC (1612.00796),
s1 (2501.19393), DeepSeek-R1 (2501.12948).
