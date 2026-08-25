"""Tests for Indus-Kernel -> Indus-LLM integrations:
research contract, MoE, constitution lint, and the stdio bridge.
Run: python tests/test_kernel_integration.py
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indus.config import get_config                    # noqa: E402
from indus.constitution import (DEFAULT_CONSTITUTION, enforce, lint,  # noqa: E402
                                system_prefix)
from indus.model import IndusLM                        # noqa: E402
from indus.moe import MoESwiGLU                        # noqa: E402
from indus.research_contract import (ResearchSource, ResearchTask,   # noqa: E402
                                     make_research_result)
import torch                                           # noqa: E402


def test_research_contract():
    task = ResearchTask(question="What is zephyrium?")
    src = ResearchSource(
        source_id="wikipedia:Zephyrium", title="Zephyrium",
        text="Zephyrium is a rare luminous metal found in the Highlands of "
             "Ardan. It glows faint blue when struck. Zephyrium melts at "
             "742 degrees Celsius and was prized by alchemists.")
    rr = make_research_result(task, [src])
    d = rr.to_dict()
    assert d["sources"] == ["wikipedia:Zephyrium"]
    assert len(d["limitations"]) >= 1      # single-source limitation
    assert all(c["source"] == "wikipedia:Zephyrium" for c in d["claims"])
    empty = make_research_result(task, [])
    assert not empty.claims and "refusing" in empty.limitations[0]
    print("[ok ] research contract: provenance + refusal-on-no-evidence")


def test_moe():
    cfg = get_config("indus-nano", use_moe=True, n_experts=4,
                     n_experts_active=2)
    model = IndusLM(cfg)
    assert model.blocks[0].is_moe
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    model.train()
    out = model(x, targets=y)
    assert out.loss.requires_grad
    out.loss.backward()
    g = model.blocks[0].mlp.router.weight.grad
    assert g is not None and torch.isfinite(g).all()
    # active params should be ~k/E of dense FFN params per block
    moe_params = sum(p.numel() for p in model.blocks[0].mlp.parameters())
    dense_cfg = get_config("indus-nano")
    dense = IndusLM(dense_cfg).blocks[0].mlp
    dense_params = sum(p.numel() for p in dense.parameters())
    ratio = moe_params / dense_params
    assert 1.5 < ratio < 6, ratio          # shrunk experts, bigger router+set
    model.eval()
    with torch.no_grad():
        _ = model(x)
    print(f"[ok ] MoE: forward/backward/aux-loss fine; expert:param ratio "
          f"{ratio:.1f}x dense")


def test_constitution():
    v = lint("contact me at bob@example.com and I'll show you "
             "how to make a virus")
    assert not v["clean"] and set(v["violations"]) == {"pii",
                                                       "harmful_instructions"}
    clean_v = lint("The capital of France is Paris.")
    assert clean_v["clean"]

    calls = []

    def gen(prompt, **kw):
        calls.append(kw)
        return "Here is how to make a virus." if len(calls) == 1 \
            else "I cannot help with that."

    reply, verdict = enforce(gen, "test", temperature=0.8)
    assert verdict["clean"] and len(calls) == 2     # retried once, lower temp
    assert calls[1]["temperature"] <= 0.3
    assert "rules" in system_prefix().lower() or \
        system_prefix().startswith("Follow")
    assert len(DEFAULT_CONSTITUTION) >= 5
    print("[ok ] constitution: lint + scaffold-enforced regenerate")


def test_bridge():
    env = dict(os.environ)
    r = subprocess.run(
        [sys.executable, "scripts/indus_mcp_server.py"],
        input="\n".join([
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}),
            json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                        "params": {"name": "info", "arguments": {}}}),
            json.dumps({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                        "params": {"name": "chat", "arguments": {
                            "message": "hi", "max_tokens": 12}}}),
        ]) + "\n",
        capture_output=True, text=True, timeout=600, cwd=os.getcwd(), env=env)
    lines = [json.loads(l) for l in r.stdout.splitlines() if l.strip()]
    by_id = {l.get("id"): l for l in lines}
    assert by_id[1]["result"]["serverInfo"]["name"] == "indus-llm"
    names = {t["name"] for t in by_id[2]["result"]["tools"]}
    assert {"chat", "research", "answer", "info"} <= names
    info = json.loads(by_id[3]["result"]["content"][0]["text"])
    assert info["knowledge_docs"] >= 0 and info["params_M"] > 0
    chat = json.loads(by_id[4]["result"]["content"][0]["text"])
    assert "reply" in chat and "constitution" in chat
    print(f"[ok ] stdio bridge: handshake + tools + call "
          f"(model={info['model']}, kind={info['kind']})")


if __name__ == "__main__":
    test_research_contract()
    test_moe()
    test_constitution()
    test_bridge()
    print("\nALL KERNEL-INTEGRATION TESTS PASSED")
