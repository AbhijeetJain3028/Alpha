"""Indus: a decoder-only transformer language model, built from scratch in PyTorch.

Architecture (modern LLaMA/GPT-style):
    - Token embeddings only; positions handled by Rotary Embeddings (RoPE)
    - Pre-norm with RMSNorm (no LayerNorm, no biases anywhere)
    - Multi-head causal self-attention with Grouped-Query Attention (GQA)
    - SwiGLU feed-forward network
    - Tied input embedding / output projection
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import IndusConfig


# --------------------------------------------------------------------------- RoPE
def build_rope_cache(head_dim: int, max_seq_len: int, base: float = 10000.0,
                     device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables of shape (max_seq_len, head_dim)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)                    # (T, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)             # (T, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor,
               sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: (B, n_head, T, head_dim); cos, sin: (T, head_dim)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


# ------------------------------------------------------------------------- blocks
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class CausalSelfAttention(nn.Module):
    """Multi-head attention with RoPE and grouped-query attention."""

    def __init__(self, config: IndusConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        assert config.n_head % config.n_kv_head == 0
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.group_size = config.n_head // config.n_kv_head

        # fused QKV projection (no bias)
        self.c_attn = nn.Linear(config.n_embd,
                                (config.n_head + 2 * config.n_kv_head) * self.head_dim,
                                bias=False)
        self.c_proj = nn.Linear(config.n_head * self.head_dim, config.n_embd, bias=False)
        self.attn_dropout = config.dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split([self.n_head * self.head_dim,
                             self.n_kv_head * self.head_dim,
                             self.n_kv_head * self.head_dim], dim=2)
        # (B, T, C) -> (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        if self.n_kv_head != self.n_head:  # GQA: share kv heads across query groups
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)

        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        return self.c_proj(y)


class SwiGLU(nn.Module):
    """Gated feed-forward network: down( silu(gate(x)) * up(x) )."""

    def __init__(self, config: IndusConfig):
        super().__init__()
        hidden = int(8 * config.n_embd / 3)          # ~4x params like GELU MLP
        hidden = ((hidden + config.ffn_multiple_of - 1)
                  // config.ffn_multiple_of) * config.ffn_multiple_of
        self.w_gate = nn.Linear(config.n_embd, hidden, bias=False)
        self.w_up = nn.Linear(config.n_embd, hidden, bias=False)
        self.w_down = nn.Linear(hidden, config.n_embd, bias=False)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.dropout(
            self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)),
            p=self.dropout, training=self.training)


class Block(nn.Module):
    def __init__(self, config: IndusConfig):
        super().__init__()
        self.norm_1 = RMSNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.norm_2 = RMSNorm(config.n_embd)
        if getattr(config, "use_moe", False) and config.n_experts > 1:
            from .moe import MoESwiGLU
            self.mlp = MoESwiGLU(config)
            self.is_moe = True
        else:
            self.mlp = SwiGLU(config)
            self.is_moe = False

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_1(x), cos, sin)
        x = x + self.mlp(self.norm_2(x))
        return x


# ---------------------------------------------------------------------------- LM
@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class IndusLM(nn.Module):
    """The complete Indus language model."""

    def __init__(self, config: IndusConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm_f = RMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying
        self.lm_head.weight = self.tok_emb.weight

        cos, sin = build_rope_cache(config.head_dim, config.block_size,
                                    config.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # scaled init on residual projections (nanoGPT trick for stability)
        for pn, p in self.named_parameters():
            if pn.endswith(("c_proj.weight", "w_down.weight")):
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()   # lm_head shares this storage
        return n

    def forward(self, idx: torch.Tensor,
                targets: torch.Tensor | None = None) -> ModelOutput:
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"sequence length {T} exceeds block size {self.config.block_size}"

        cos = self.rope_cos[:T].view(1, 1, T, -1)
        sin = self.rope_sin[:T].view(1, 1, T, -1)

        x = self.tok_emb(idx)
        x = self.dropout(x)
        aux = 0.0
        for block in self.blocks:
            x = block(x, cos, sin)
            if getattr(block, "is_moe", False):
                aux = aux + block.mlp.aux_loss
        x = self.norm_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-100)
            if getattr(self.config, "use_moe", False) and self.training:
                n_moe = sum(1 for b in self.blocks if b.is_moe)
                if n_moe:
                    loss = loss + 0.01 * aux / n_moe
        else:
            logits = self.lm_head(x)
            loss = None
        return ModelOutput(logits=logits, loss=loss)

    # ------------------------------------------------------------- generation
    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 0.8, top_k: int | None = 50,
                 endoftext_id: int | None = None) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                else idx[:, -self.config.block_size:]
            out = self(idx_cond)
            logits = out.logits[:, -1, :]

            if temperature <= 1e-6:                      # greedy
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    thresh = torch.topk(logits, k, dim=-1).values[..., -1, None]
                    logits = logits.masked_fill(logits < thresh, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, next_id), dim=1)
            if endoftext_id is not None and (next_id == endoftext_id).all():
                break
        return idx


def create_model(config: IndusConfig, device: str = "cpu") -> IndusLM:
    model = IndusLM(config)
    n_params = model.num_params()
    print(f"Indus ({config.name}): {n_params / 1e6:.2f}M parameters "
          f"({model.num_params(non_embedding=False) / 1e6:.2f}M with embeddings)")
    return model.to(device)


@torch.no_grad()
def ensure_vocab_size(model: IndusLM, new_vocab: int) -> bool:
    """Grow the embedding table to new_vocab rows deterministically.

    New special-token rows (chat markers etc.) are initialized by copying
    the <|endoftext|> row instead of leaving random init - random rows poison
    generation when a pretrained ckpt is loaded with an enlarged tokenizer.
    Keeps input/output tying intact. Returns True if resized.
    """
    cur = model.config.vocab_size
    if new_vocab <= cur:
        return False
    w = model.tok_emb.weight.data
    pad = w[cur - 1:cur].expand(new_vocab - cur, -1).clone()
    model.tok_emb.weight = nn.Parameter(torch.cat([w, pad], dim=0))
    model.lm_head.weight = model.tok_emb.weight      # preserve weight tying
    model.config.vocab_size = new_vocab
    return True
