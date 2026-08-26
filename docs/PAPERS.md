# Research Bibliography — Building Indus

Every paper that informs a design decision in this repository, organized by
component. Each entry states exactly what Indus takes from it.

## 1. Foundations

| Paper | Venue | What Indus uses |
|---|---|---|
| **Attention Is All You Need** — Vaswani et al., 2017 ([arXiv:1706.03762](https://arxiv.org/abs/1706.03762)) | NeurIPS | Decoder-only transformer; residual stream; causal self-attention |
| **Layer Normalization** — Ba, Kiros, Hinton, 2016 ([arXiv:1607.06450](https://arxiv.org/abs/1607.06450)) | — | Baseline norm (superseded by RMSNorm) |
| **On Layer Normalization in the Transformer Architecture** — Xiong et al., 2020 ([arXiv:2002.04745](https://arxiv.org/abs/2002.04745)) | ICML | Pre-norm placement (norm inside residual branch) → stable deep training without warmup tricks |

## 2. Positional encoding

| Paper | Venue | What Indus uses |
|---|---|---|
| **RoFormer: Enhanced Transformer with Rotary Position Embedding** — Su et al., 2021 ([arXiv:2104.09864](https://arxiv.org/abs/2104.09864)) | Neurocomputing | **RoPE**: rotates q/k by position-dependent angles; relative-position awareness baked into attention (`model.py: build_rope_cache`, `apply_rope`) |

## 3. Attention efficiency

| Paper | Venue | What Indus uses |
|---|---|---|
| **Fast Transformer Decoding (MQA)** — Shazeer, 2019 ([arXiv:1911.02150](https://arxiv.org/abs/1911.02150)) | — | Idea of sharing KV heads across queries |
| **GQA: Training Generalized Multi-Query Transformer Models** — Ainslie et al., 2023 ([arXiv:2305.13245](https://arxiv.org/abs/2305.13245)) | EMNLP | **Grouped-Query Attention** (`n_kv_head < n_head`): ~40% fewer attention params/memory at near-MHA quality (`config.py` presets use ratios from the paper) |
| **FlashAttention** — Dao et al., 2022 ([arXiv:2205.14135](https://arxiv.org/abs/2205.14135)) | NeurIPS | Exact fast attention via `F.scaled_dot_product_attention` — IO-aware fused kernels with no explicit T×T mask materialization |
| **FlashAttention-2** — Dao, 2023 ([arXiv:2307.08691](https://arxiv.org/abs/2307.08691)) | — | (Inherited automatically by newer torch builds) |

## 4. Feed-forward & normalization

| Paper | Venue | What Indus uses |
|---|---|---|
| **GLU Variants Improve Transformer** — Shazeer, 2020 ([arXiv:2002.05202](https://arxiv.org/abs/2002.05202)) | — | **SwiGLU** FFN: `down(silu(gate(x)) * up(x))`; hidden ≈ 8d/3 to keep param parity with GELU MLPs (`model.py: SwiGLU`) |
| **Root Mean Square Layer Normalization** — Zhang & Sennrich, 2019 ([arXiv:1910.07467](https://arxiv.org/abs/1910.07467)) | NeurIPS | **RMSNorm** (no mean-centering, no bias) → same quality, faster (`model.py: RMSNorm`) |
| **PaLM: Scaling Language Modeling with Pathways** — Chowdhery et al., 2022 ([arXiv:2204.02311](https://arxiv.org/abs/2204.02311)) | JMLR | Validation of SwiGLU+RMSNorm+parallel-residual choices at scale; no-bias Linear layers |

## 5. Tokenization

| Paper | Venue | What Indus uses |
|---|---|---|
| **Neural Machine Translation of Rare Words with Subword Units (BPE)** — Sennrich et al., 2016 ([arXiv:1508.07909](https://arxiv.org/abs/1508.07909)) | ACL | Byte-pair merge objective |
| **GPT-2 / Language Models are Unsupervised Multitask Learners** — Radford et al., 2019 ([paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)) | — | **Byte-level BPE**: no OOV ever; word-level pre-tokenization regex before counting merges (`tokenizer.py`) |

## 6. Training dynamics & optimization

| Paper | Venue | What Indus uses |
|---|---|---|
| **Decoupled Weight Decay Regularization** — Loshchilov & Hutter, 2017 ([arXiv:1711.05101](https://arxiv.org/abs/1711.05101)) | ICLR | **AdamW** (decoupled WD), β₂=0.95 for LM training |
| **SGDR: Stochastic Gradient Descent with Warm Restarts** — Loshchilov & Hutter, 2016 ([arXiv:1608.03983](https://arxiv.org/abs/1608.03983)) | ICLR | Cosine LR decay to 10% of peak |
| **Accurate, Large Minibatch SGD** — Goyal et al., 2017 ([arXiv:1706.02677](https://arxiv.org/abs/1706.02677)) | — | Linear LR warmup schedule |
| **Mixed Precision Training** — Micikevicius et al., 2017 ([arXiv:1710.03740](https://arxiv.org/abs/1710.03740)) | ICLR | autocast fp16 (+GradScaler on pre-Ampere GPUs like Kaggle P100/T4); bf16 on Ampere+ |
| **Tensor Programs V: μTransfer / scaling init** — Yang et al., 2021 ([arXiv:2103.10365](https://arxiv.org/abs/2103.10365)) | — | Residual-projection init scaled by 1/√(2·n_layer) (nanoGPT-style stability trick) |
| **Using the Output Embedding to Improve Language Models (weight tying)** — Press & Wolf, 2016 ([arXiv:1608.05859](https://arxiv.org/abs/1608.05859)) | EACL | Tied input/output embeddings |

## 7. Scale, data recipes & open datasets

| Paper / Asset | Venue | What Indus uses |
|---|---|---|
| **Scaling Laws for Neural Language Models** — Kaplan et al., 2020 ([arXiv:2001.08361](https://arxiv.org/abs/2001.08361)) | — | Loss-vs-compute shape; why bigger data beats bigger models at fixed budget |
| **Training Compute-Optimal Large Language Models (Chinchilla)** — Hoffmann et al., 2022 ([arXiv:2203.15556](https://arxiv.org/abs/2203.15556)) | NeurIPS | ~20 tokens-per-parameter target → `indus-base` (~38M) trains ~800M tokens |
| **TinyStories** — Eldan & Li, 2023 ([arXiv:2305.07759](https://arxiv.org/abs/2305.07759)) | — | Corpus source #1 (CDLA-Sharing-1.0): narrative fluency at tiny scale |
| **The FineWeb Datasets** — Penedo et al., 2024 ([arXiv:2406.17557](https://arxiv.org/abs/2406.17557)) | — | Corpus source #2 (ODC-BY-1.0): classifier-filtered educational web text (FineWeb-Edu sample) |
| **WikiText-103** — Merity et al., 2016 ([arXiv:1609.07843](https://arxiv.org/abs/1609.07843)) | — | Corpus source #3 + eval PPL (CC BY-SA 3.0) |
| **SmolLM2: When Smol Goes Big** — Allal et al., 2025 ([arXiv:2502.02737](https://arxiv.org/abs/2502.02737)) | — | The definitive small-model recipe Indus follows for its scale class: single-stage high-quality mix (web-edu + wiki), GQA, over-training small models; benchmark selection (LAMBADA/ARC/PPL) |
| **Cosmopedia** — Ben Allal et al., 2024 ([arXiv:2402.14640](https://arxiv.org/abs/2402.14640)) | — | Reference synthetic-data approach (Apache-2.0); candidate future corpus slice |
| **LLaMA: Open and Efficient Foundation Language Models** — Touvron et al., 2023 ([arXiv:2302.13971](https://arxiv.org/abs/2302.13971)) | — | The exact architecture stack Indus implements: RoPE + SwiGLU + RMSNorm + no biases |
| **nanoGPT** — Karpathy, 2023 ([github](https://github.com/karpathy/nanoGPT)) | — | Memmap uint16 token loading, random-offset batch sampling, scaled-init, training loop ergonomics |

### SFT / alignment stage

| Paper / Dataset | License | Role |
|---|---|---|
| **OpenAssistant Conversations (OASST1)** — Köpf et al., 2023 ([arXiv:2304.07327](https://arxiv.org/abs/2304.07327)) | Apache-2.0 | Human-written multi-turn chat trees → SFT conversations |
| **Free Dolly (databricks-dolly-15k)** — Conover et al., 2023 ([blog](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm)) | CC-BY-SA-3.0 | Human instruction/response pairs |
| **SmolTalk** — Allal et al., 2024 ([smollm/corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus)) | Apache-2.0 | Everyday-conversations slice; SmolLM-Instruct's SFT backbone |
| **InstructGPT** — Ouyang et al., 2022 ([arXiv:2203.02155](https://arxiv.org/abs/2203.02155)) | NeurIPS | Roadmap: SFT → RLHF after pretraining |
| **DPO** — Rafailov et al., 2023 ([arXiv:2305.18290](https://arxiv.org/abs/2305.18290)) | NeurIPS | Simpler preference tuning path |

### Notes on other open model families surveyed
- **NVIDIA Nemotron family** (e.g., Llama-Nemotron / Nemotron-4): NVIDIA publishes open post-training
  datasets (CC-BY-4.0) and model recipes; their architectural core (RoPE/GQA/SwiGLU decoder) matches
  what Indus already implements. No proprietary component was imported; ideas only.
- **Qwen / Phi / Gemma small variants**: confirm the same modern stack; licenses are custom
  (research-restricted or terms-of-use bound), so no weights or data were taken from them.
- **lm-evaluation-harness**: task definitions informed Indus's compact evaluator
  (`scripts/evaluate.py` reimplements LAMBADA + ARC-Easy scoring from scratch).

## 8. Autonomy engine (`indus/autonomous.py`, `scripts/auto_learn.py`)

| Paper | Venue | What Indus uses |
|---|---|---|
| **REALM: Retrieval-Augmented LM** — Guu et al., 2020 ([arXiv:2002.08909](https://arxiv.org/abs/2002.08909)) | ICML | Doctrine: knowledge lives in a retriever, not the network |
| **RAG** — Lewis et al., 2020 ([arXiv:2005.11401](https://arxiv.org/abs/2005.11401)) | NeurIPS | Grounded generation from retrieved passages |
| **RA-DIT** — Izacard et al., 2022 ([arXiv:2212.10517](https://arxiv.org/abs/2212.10517)) | — | Retro-fit LM via retrieval-grounded fine-tuning |
| **Toolformer** — Schick et al., 2023 ([arXiv:2302.04761](https://arxiv.org/abs/2302.04761)) | NeurIPS | Scaffold-shaped tool calls; degraded-but-deterministic at tiny scale |
| **STaR** — Zelikman et al., 2022 ([arXiv:2203.14465](https://arxiv.org/abs/2203.14465)) | NeurIPS | Keep-only-winners self-training loop |
| **Self-RAG** — Asai et al., 2023 ([arXiv:2310.11511](https://arxiv.org/abs/2310.11511)) | ICLR | Critique-gated retrieval tokens → our citation linting |
| **WebGPT** — Nakano et al., 2021 ([arXiv:2112.09332](https://arxiv.org/abs/2112.09332)) | — | Search-augmented QA with source citations |
| **Search-R1** — Jin et al., 2025 ([arXiv:2503.09516](https://arxiv.org/abs/2503.09516)) | — | Evidence that RL search policy needs ≥3B → tiny models use scaffolded search instead |
| **EWC** — Kirkpatrick et al., 2017 ([arXiv:1612.00796](https://arxiv.org/abs/1612.00796)) | PNAS | Catastrophic-forgetting defense → our eval-gate + replay-buffer rollback |

## 9. Kernel integrations (from Indus-Kernel `ik_*`)

| Source | What was ported | Where it lives now |
|---|---|---|
| `ik_research` provenance contract | ResearchTask/Source/Claim/Result; never invent sources; refuse without evidence | `indus/research_contract.py` |
| `ik_indus_llm/moe.py` | Sparse MoE-SwiGLU experts + load-balancing aux loss (k/E-shrunk for FLOP parity) | `indus/moe.py` + `config.use_moe` (dormant until next pretrain) |
| `ik_indus_llm/constitution.py` | Generation-time constitution: principles prefix + deterministic PII/harm lint + single low-temp regenerate | `indus/constitution.py` |
| A2A/MCP ABI | stdio JSON-RPC tool server (`chat`/`research`/`answer`/`info`) so all 40 kernel subsystems can drive Indus | `scripts/indus_mcp_server.py` |

Supporting papers: **Constitutional AI** — Bai et al., 2022 ([arXiv:2212.08073](https://arxiv.org/abs/2212.08073)); **MoE** — Shazeer et al., 2017 ([arXiv:1701.06538](https://arxiv.org/abs/1701.06538)); **Switch Transformer** — Fedus et al., 2022 ([arXiv:2101.03961](https://arxiv.org/abs/2101.03961)); **DeepSeekMoE** — Dai et al., 2024 ([arXiv:2401.06066](https://arxiv.org/abs/2401.06066)).

## 10. Efficiency roadmap (from the 2026 playbook sweep)

MobileLLM deep-narrow sizing ([arXiv:2402.14905](https://arxiv.org/abs/2402.14905)) · WSD schedule with high-quality cooldown / MiniCPM ([arXiv:2404.06395](https://arxiv.org/abs/2404.06395)) · Muon optimizer (Jordan 2024; Kimi K2 [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)) · Model Soups ([arXiv:2203.05482](https://arxiv.org/abs/2203.05482)) · multi-token prediction ([arXiv:2404.19737](https://arxiv.org/abs/2404.19737)). Full strategy detail: [PLAYBOOK.md](PLAYBOOK.md).

## Mapping summary

```
Indus file          ← papers
─────────────────────────────────────────────────────
tokenizer.py        ← Sennrich'16, Radford'19
model.py:RoPE       ← Su'21
model.py:RMSNorm    ← Zhang&Sennrich'19
model.py:SwiGLU     ← Shazeer'20
model.py:GQA        ← Shazeer'19, Ainslie'23
model.py:SDPA       ← Dao'22
model.py:tied emb   ← Press&Wolf'16
train.py:AdamW      ← Loshchilov&Hutter'17/19
train.py:cosine     ← SGDR'16
data.py             ← nanoGPT, Kaplan'20
presets/tokens      ← Chinchilla'22, TinyStories'23
whole arch          ← Vaswani'17 decoder + Xiong'20 pre-norm + LLaMA'23
muon.py             ← Jordan'24 Muon + Kimi K2'25 scale validation
export_gguf.py      ← LLaMA'23 tensor ABI (llama.cpp/GGUF ecosystem)
autonomous.py       ← REALM/RAG/RA-DIT + Toolformer + STaR/Self-RAG +
                      WebGPT/Search-R1 + EWC (replay+gate)
moe.py              ← Shazeer'17 / Switch'22 / DeepSeekMoE'24
constitution.py     ← Bai'22 Constitutional AI
research_contract   ← ik_research provenance doctrine
indus_mcp_server    ← A2A v1.0 / MCP ABI bridge
```
