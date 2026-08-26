#!/usr/bin/env python3
"""Indus distillation kernel - data-distillation from an open teacher.

Doctrine (playbook §6): at tiny scale, DATA distillation beats logit-KD -
it is tokenizer-agnostic, license-clean (Apache-2.0 teachers only), and
needs zero custom training loops. The teacher never touches Indus weights;
it only writes text that becomes pretraining/SFT calories.

Pipeline (runs on one Kaggle T4):
  1. load teacher (SmolLM2-360M-Instruct default; Qwen2.5-0.5B-Instruct alt)
  2. build a prompt pool: instruction seeds + knowledge-store questions +
     story/story continuations (covers chat, facts, narrative registers)
  3. batch-generate completions (fp16, top-p 0.9, dedup by 3-gram overlap)
  4. write distill.jsonl {source, prompt, completion} -> upload to the Hub

Placeholders __HF_TOKEN__ / __HF_REPO_ID__ / __TEACHER__ / __N_SAMPLES__
are replaced by scripts/kaggle_run.py render-distill at push time.
"""

import glob
import hashlib
import json
import os
import sys

INPUT = os.environ.get("INDUS_INPUT", "/kaggle/input")
WORK = os.environ.get("INDUS_WORK", "/kaggle/working")
os.makedirs(WORK, exist_ok=True)

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        HF_TOKEN = ""
if not HF_TOKEN:
    HF_TOKEN = "__HF_TOKEN__"
HF_REPO_ID = "__HF_REPO_ID__"
TEACHER = "__TEACHER__"
N_SAMPLES = int("__N_SAMPLES__")

# ------------------------------------------------------------------ prompts
CHAT_SEEDS = [
    "Explain {} in one short paragraph for a curious child.",
    "Write three sentences about {}.",
    "What is {}? Answer briefly.",
    "Give two fun facts about {}.",
]
STORY_SEEDS = [
    "Once upon a time,",
    "The little robot looked up at the sky and",
    "Maya found a door in the garden that",
]
FACT_TOPICS = [
    "the water cycle", "why the sky is blue", "how plants eat sunlight",
    "the Eiffel Tower", "photosynthesis", "gravity", "the solar system",
    "bees and flowers", "recycling", "volcanoes", "dinosaurs",
    "the internet", "vaccines", "electricity", "rainbows",
]


def build_prompts() -> list[str]:
    import random
    rng = random.Random(1337)
    prompts = []
    for topic in FACT_TOPICS:
        for tmpl in CHAT_SEEDS:
            prompts.append(tmpl.format(topic))
    while len(prompts) < N_SAMPLES:
        prompts.append(rng.choice(STORY_SEEDS))
    return prompts[:N_SAMPLES]


# ------------------------------------------------------------------ teacher
def main() -> None:
    from huggingface_hub import HfApi
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("[setup] teacher:", TEACHER, "| torch:", torch.__version__,
          "| cuda:", torch.cuda.is_available())
    tok = AutoTokenizer.from_pretrained(TEACHER)
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER, torch_dtype=torch.float16).to("cuda").eval()

    prompts = build_prompts()
    print(f"[gen ] {len(prompts)} prompts")

    seen = set()
    out_path = os.path.join(WORK, "distill.jsonl")
    n_written = 0
    B = 16
    with torch.no_grad():
        for i in range(0, len(prompts), B):
            batch = prompts[i:i + B]
            texts = [tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True) for p in batch]
            enc = tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to("cuda")
            gen = model.generate(
                **enc, max_new_tokens=160, do_sample=True,
                top_p=0.9, temperature=0.8,
                pad_token_id=tok.eos_token_id)
            for j, p in enumerate(batch):
                new = gen[j][enc["input_ids"].shape[1]:]
                comp = tok.decode(new, skip_special_tokens=True).strip()
                if len(comp) < 60:
                    continue
                key = hashlib.md5(comp.lower().encode()).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"source": f"distill:{TEACHER}",
                                        "prompt": p,
                                        "completion": comp}) + "\n")
                n_written += 1
            if (i // B) % 10 == 0:
                print(f"  {i + len(batch)}/{len(prompts)} | kept {n_written}",
                      flush=True)

    print(f"[done] {n_written} distilled samples -> {out_path}")
    api = HfApi(token=HF_TOKEN or None)
    try:
        api.upload_file(path_or_fileobj=out_path, path_in_repo="distill.jsonl",
                        repo_id=HF_REPO_ID, repo_type="model",
                        commit_message=f"data-distill: {n_written} samples "
                                       f"from {TEACHER}")
        print("[hub ] uploaded distill.jsonl")
    except Exception as e:
        print("[hub ] upload failed:", e)


if __name__ == "__main__":
    main()
