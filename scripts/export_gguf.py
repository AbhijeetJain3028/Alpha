#!/usr/bin/env python3
"""Export an Indus checkpoint to Llama-compatible HF format -> GGUF-ready.

Route (playbook §5): emit standard `LlamaForCausalLM` layout
(config.json + weights) with Llama-style tensor names, so llama.cpp's
`convert_hf_to_gguf.py`, lm-eval-harness, TRL, and every Llama-compatible
tool work out of the box.

    python scripts/export_gguf.py --ckpt checkpoints/ckpt-best.pt --out out/gguf
    # then, where llama.cpp is built:
    #   python convert_hf_to_gguf.py out/gguf --outfile indus-f16.gguf --outtype f16
    #   llama-quantize indus-f16.gguf indus-q8_0.gguf Q8_0

Fused QKV is split into separate q/k/v projections; the tied lm_head is
omitted (config tie_word_embeddings=true). MoE checkpoints are rejected
(no llama arch equivalent yet).
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indus.config import IndusConfig                                  # noqa: E402


def ffn_hidden(cfg: IndusConfig) -> int:
    hidden = int(8 * cfg.n_embd / 3)
    return ((hidden + cfg.ffn_multiple_of - 1)
            // cfg.ffn_multiple_of) * cfg.ffn_multiple_of


def convert(ckpt_path: str, out_dir: str) -> None:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = IndusConfig.from_dict(ck["config"])
    if getattr(cfg, "use_moe", False):
        raise SystemExit("MoE checkpoints have no llama arch map yet")
    sd = {k: v.float() for k, v in ck["model"].items()}

    hd = cfg.head_dim
    os.makedirs(out_dir, exist_ok=True)
    out: dict[str, torch.Tensor] = {}
    out["model.embed_tokens.weight"] = sd["tok_emb.weight"]

    for i in range(cfg.n_layer):
        p, q = f"blocks.{i}.", f"model.layers.{i}."
        out[q + "input_layernorm.weight"] = sd[p + "norm_1.weight"]
        # fused QKV -> split rows [Q | K | V]
        w = sd[p + "attn.c_attn.weight"]
        nq, nkv = cfg.n_head * hd, cfg.n_kv_head * hd
        out[q + "self_attn.q_proj.weight"] = w[:nq]
        out[q + "self_attn.k_proj.weight"] = w[nq:nq + nkv]
        out[q + "self_attn.v_proj.weight"] = w[nq + nkv:]
        out[q + "self_attn.o_proj.weight"] = sd[p + "attn.c_proj.weight"]
        out[q + "post_attention_layernorm.weight"] = sd[p + "norm_2.weight"]
        out[q + "mlp.gate_proj.weight"] = sd[p + "mlp.w_gate.weight"]
        out[q + "mlp.up_proj.weight"] = sd[p + "mlp.w_up.weight"]
        out[q + "mlp.down_proj.weight"] = sd[p + "mlp.w_down.weight"]
    out["model.norm.weight"] = sd["norm_f.weight"]

    llama_cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": int(cfg.vocab_size),
        "hidden_size": int(cfg.n_embd),
        "intermediate_size": ffn_hidden(cfg),
        "num_hidden_layers": int(cfg.n_layer),
        "num_attention_heads": int(cfg.n_head),
        "num_key_value_heads": int(cfg.n_kv_head),
        "head_dim": int(hd),
        "max_position_embeddings": int(cfg.block_size),
        "rms_norm_eps": 1e-6,
        "rope_theta": float(cfg.rope_base),
        "rope_scaling": None,
        "tie_word_embeddings": True,
        "torch_dtype": "float16",
        "bos_token_id": int(cfg.vocab_size) - 1,       # <|endoftext|>
        "eos_token_id": int(cfg.vocab_size) - 1,
    }
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(llama_cfg, f, indent=2)

    try:
        from safetensors.torch import save_file
        save_file({k: v.half().contiguous()
                   for k, v in out.items()},
                  os.path.join(out_dir, "model.safetensors"))
        weight_file = "model.safetensors"
    except ImportError:
        torch.save({k: v.half() for k, v in out.items()},
                   os.path.join(out_dir, "pytorch_model.bin"))
        weight_file = "pytorch_model.bin"

    # tokenizer: copy ours alongside (byte-BPE v2 json) for reference
    tok_src = None
    for cand in ("data/tokenizer.json", "data_v2/tokenizer.json",
                 os.path.join(os.path.dirname(ckpt_path), "tokenizer.json")):
        if os.path.exists(cand):
            tok_src = cand
            break
    if tok_src:
        import shutil
        shutil.copy2(tok_src, os.path.join(out_dir, "indus_tokenizer.json"))

    n = sum(v.numel() for v in out.values())
    print(f"[done] {out_dir}/ ({weight_file}, {n / 1e6:.1f}M params in fp16)")
    print("next steps:")
    print("  python convert_hf_to_gguf.py "
          f"{out_dir} --outfile indus-f16.gguf --outtype f16")
    print("  llama-quantize indus-f16.gguf indus-q8_0.gguf Q8_0")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    convert(args.ckpt, args.out)


if __name__ == "__main__":
    main()
