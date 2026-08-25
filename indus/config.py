"""Model configuration for Indus.

A single dataclass describes every hyperparameter. Presets range from a
tiny model trainable on a laptop CPU to GPT-2-class sizes for GPUs.
"""

from dataclasses import dataclass, asdict


@dataclass
class IndusConfig:
    # architecture
    name: str = "indus-tiny"
    vocab_size: int = 4096          # includes 256 byte tokens + merges + <|endoftext|>
    block_size: int = 512           # max context length
    n_layer: int = 6                # transformer blocks
    n_head: int = 6                 # query heads
    n_kv_head: int = 6              # key/value heads (GQA when < n_head)
    n_embd: int = 384               # embedding / residual width
    ffn_multiple_of: int = 32       # SwiGLU hidden dim is rounded to this
    rope_base: float = 10000.0      # RoPE theta

    # regularization / init
    dropout: float = 0.0            # dropout on attn output + mlp output
    weight_decay: float = 0.1

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_head % self.n_kv_head == 0, "n_head must be divisible by n_kv_head"

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


# ---------------------------------------------------------------------------
# Presets. All share the same modern decoder-only design:
# RMSNorm pre-norm | Rotary position embeddings (RoPE) | SwiGLU MLP |
# Grouped-query attention | No biases | Tied input/output embeddings.
# ---------------------------------------------------------------------------
PRESETS = {
    # ~1.3M params - trains in minutes on CPU; good for tests & debugging
    "indus-nano": dict(
        name="indus-nano", vocab_size=4096, block_size=256,
        n_layer=4, n_head=4, n_kv_head=2, n_embd=128,
        ffn_multiple_of=32,
    ),
    # ~11M params - trainable on CPU overnight or GPU in minutes
    "indus-tiny": dict(
        name="indus-tiny", vocab_size=4096, block_size=512,
        n_layer=6, n_head=6, n_kv_head=3, n_embd=384,
        ffn_multiple_of=64,
    ),
    # ~38M params - GQA small-base; the main free-GPU training target
    "indus-base": dict(
        name="indus-base", vocab_size=16384, block_size=1024,
        n_layer=10, n_head=8, n_kv_head=4, n_embd=512,
        ffn_multiple_of=64, rope_base=10000.0,
    ),
    # ~42M params - needs a small GPU for reasonable training time
    "indus-small": dict(
        name="indus-small", vocab_size=8192, block_size=1024,
        n_layer=12, n_head=12, n_kv_head=4, n_embd=576,
        ffn_multiple_of=64, rope_base=10000.0,
    ),
    # ~150M params - GPT-2 class, GPU required
    "indus-medium": dict(
        name="indus-medium", vocab_size=16384, block_size=2048,
        n_layer=16, n_head=16, n_kv_head=4, n_embd=960,
        ffn_multiple_of=128, rope_base=10000.0,
    ),
}


def get_config(preset: str | None = None, **overrides) -> IndusConfig:
    if preset is not None:
        base = dict(PRESETS[preset])
        cfg = IndusConfig(**base)
    else:
        cfg = IndusConfig()
    for k, v in overrides.items():
        if v is None:
            continue
        if not hasattr(cfg, k):
            raise AttributeError(f"Unknown config field: {k}")
        setattr(cfg, k, v)
    cfg.__post_init__()
    return cfg
