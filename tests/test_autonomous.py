"""Smoke tests for the autonomous research + self-training engine.

Run: python tests/test_autonomous.py   (or pytest tests/test_autonomous.py)
No network required: uses an offline fake corpus.
"""

import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indus.autonomous import (KnowledgeStore, SelfLearner, WebCorpus,
                              answer_grounded, autonomous_cycle,
                              grounded_prompt)
from indus.config import get_config
from indus.model import IndusLM, ensure_vocab_size
from indus.tokenizer import BPETokenizer

FAKE_DOCS = [
    {"title": "Zephyrium", "source": "wikipedia:Zephyrium",
     "text": "Zephyrium is a rare luminous metal found only in the fictional "
             "Highlands of Ardan. It glows faint blue when struck and melts "
             "at exactly 742 degrees Celsius. Alchemists once valued "
             "zephyrium more than gold for making mirrors."},
    {"title": "Ardan Highlands", "source": "wikipedia:Ardan Highlands",
     "text": "The Ardan Highlands are a fictional mountain region famous "
             "for zephyrium mines and mirror crafting. The capital city is "
             "Velmora, known for its glass observatory."},
    {"title": "Mirror crafting", "source": "web:mirror-crafting",
     "text": "Traditional mirror crafting in Ardany uses polished zephyrium "
             "backed with silver. The blue glow fades after polishing."},
]


class OfflineCorpus(WebCorpus):
    """Deterministic stand-in for web research (no network in tests)."""

    def gather(self, topic: str) -> list[dict]:
        return [d for d in FAKE_DOCS if topic.lower()[:4] in d["text"].lower()
                or topic.lower()[:4] in d["title"].lower()] or FAKE_DOCS[:1]


def main() -> None:
    tok = BPETokenizer.load(os.path.join("data", "tokenizer.json"))
    tok.add_chat_specials()

    # ---- KnowledgeStore -------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        store = KnowledgeStore(os.path.join(td, "kb.db"))
        for d in FAKE_DOCS:
            store.add(d)
        assert store.count() == 3, store.count()
        hits = store.search("zephyrium melting point", k=2)
        assert hits and "zephyrium" in hits[0]["text"].lower(), \
            f"retrieval failed: {hits}"
        print("[ok ] KnowledgeStore: add/search/bm25")

        # ---- model + learner ---------------------------------------------
        cfg = get_config("indus-nano", vocab_size=len(tok.vocab))
        model = IndusLM(cfg)
        assert ensure_vocab_size(model, len(tok.vocab)) is False
        learner = SelfLearner(model, tok, device="cpu")

        rep = learner.learn(FAKE_DOCS, steps=3, batch_size=2)
        assert set(rep) >= {"probe_before", "probe_after", "accepted"}
        assert isinstance(rep["accepted"], bool)
        print(f"[ok ] SelfLearner eval-gate report: {json.dumps(rep)[:120]}")

        # ---- grounded prompting + answering -------------------------------
        p = grounded_prompt(tok, "What is zephyrium?", FAKE_DOCS[:2])
        assert "<|system|>" in p and "[1]" in p and "[2]" in p
        reply, cites = answer_grounded(model, tok, "What is zephyrium?",
                                       FAKE_DOCS[:2], max_new_tokens=24)
        assert isinstance(reply, str) and len(cites) == 2
        print(f"[ok ] grounded answer: {reply[:60]!r} cites={cites}")

        # ---- full cycle (offline corpus) ----------------------------------
        report = autonomous_cycle(model, tok, "Zephyrium",
                                  corpus=OfflineCorpus(), store=store,
                                  learner=learner, verbose=False)
        assert report["sources_found"] >= 1
        assert report["store_size"] == 3          # dedup: re-adds are no-ops
        assert "answer" in report and report["citations"]
        lrn = report.get("learning", {})
        assert lrn["probe_before"] > 0.5          # real NLL, not a broken 0.0
        print(f"[ok ] autonomous_cycle: sources={report['sources_found']} "
              f"store={report['store_size']} accepted={lrn.get('accepted')} "
              f"probe {lrn.get('probe_before')}->{lrn.get('probe_after')}")

    print("\nALL AUTONOMY TESTS PASSED")


if __name__ == "__main__":
    main()
