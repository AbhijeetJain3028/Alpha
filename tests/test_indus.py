"""Smoke tests for the Indus package.

Run directly (no pytest needed):   python tests/test_indus.py
or via pytest:                     pytest tests/ -q
"""

import math
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indus.config import get_config, PRESETS          # noqa: E402
from indus.tokenizer import BPETokenizer              # noqa: E402
from indus.model import IndusLM                       # noqa: E402


def test_tokenizer_roundtrip():
    text = "Once upon a time, there was a little star! It twinkled at night... 123"
    tok = BPETokenizer()
    tok.train(text * 20, vocab_size=320)
    ids = tok.encode(text)
    assert all(0 <= i < len(tok.vocab) for i in ids)
    assert tok.decode(ids) == text, f"roundtrip failed: {tok.decode(ids)!r}"
    # persistence
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "tok.json")
        tok.save(p)
        tok2 = BPETokenizer.load(p)
        assert tok2.decode(tok2.encode(text)) == text
        assert len(tok2.vocab) == len(tok.vocab)
    print("tokenizer roundtrip + save/load: OK")


def test_model_forward():
    cfg = get_config("indus-nano", vocab_size=512, block_size=64,
                     n_layer=2, n_head=4, n_kv_head=2, n_embd=96)
    model = IndusLM(cfg)
    x = torch.randint(0, 512, (2, 32))
    out = model(x, targets=x)
    assert out.logits.shape == (2, 32, 512)
    assert torch.isfinite(out.loss)
    # tied embeddings give the model a self-similarity prior, so untrained
    # loss is typically somewhat below ln(vocab_size) - just check sanity
    assert 0 < out.loss.item() < math.log(512.0) + 0.5
    print(f"forward pass: OK (loss={out.loss.item():.3f}, "
          f"params={model.num_params():,})")


def test_generation_shapes():
    cfg = get_config("indus-nano", vocab_size=256, block_size=32,
                     n_layer=1, n_head=4, n_kv_head=4, n_embd=64)
    model = IndusLM(cfg)
    x = torch.zeros((1, 4), dtype=torch.long)
    y = model.generate(x, max_new_tokens=10, temperature=0.9, top_k=10)
    assert y.shape == (1, 14)
    greedy = model.generate(x, max_new_tokens=5, temperature=0.0, top_k=None)
    assert greedy.shape == (1, 9)
    print("generation: OK")


def test_gqa_matches_mha_param_count():
    """GQA should have fewer attention params than full MHA."""
    full = get_config("indus-nano", vocab_size=512, n_layer=1, n_head=4,
                      n_kv_head=4, n_embd=128)
    gqa = get_config("indus-nano", vocab_size=512, n_layer=1, n_head=4,
                     n_kv_head=2, n_embd=128)
    m_full, m_gqa = IndusLM(full), IndusLM(gqa)
    assert m_gqa.num_params(non_embedding=False) < m_full.num_params(non_embedding=False)
    print(f"GQA param reduction: OK ({m_full.num_params()} -> {m_gqa.num_params()})")


def test_training_step_decreases_loss():
    torch.manual_seed(0)
    cfg = get_config("indus-nano", vocab_size=128, block_size=16,
                     n_layer=1, n_head=2, n_kv_head=2, n_embd=48)
    model = IndusLM(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    data = torch.arange(128).repeat(100)
    first = None
    for i in range(30):
        idx = torch.randint(0, len(data) - 17, (8,))
        x = torch.stack([data[j:j + 16] for j in idx])
        loss = model(x, targets=x).loss
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.9, "loss should decrease when training"
    print(f"training step: OK (loss {first:.3f} -> {loss.item():.3f})")


def test_all_presets_instantiate():
    for name in PRESETS:
        cfg = get_config(name, block_size=64)  # small rope cache for the test
        m = IndusLM(cfg)
        del m
        print(f"preset {name}: instantiates OK")


if __name__ == "__main__":
    test_tokenizer_roundtrip()
    test_model_forward()
    test_generation_shapes()
    test_gqa_matches_mha_param_count()
    test_training_step_decreases_loss()
    test_all_presets_instantiate()
    print("\nall tests passed ✔")
