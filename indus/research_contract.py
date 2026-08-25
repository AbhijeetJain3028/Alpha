"""Auditable research contract - ported from Indus-Kernel `ik_research`.

Integration doctrine: the kernel's contract (never invent sources; every
claim tied to provenance; explicit limitations) merged with Indus-LLM's
working autonomy engine (`indus.autonomous`). Every autonomous_cycle report
can now be rendered as a ResearchResult that an auditor - or the Indus
Kernel control plane - can verify.

Zero dependencies beyond stdlib.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ------------------------------------------------------------------ contract
@dataclass(frozen=True)
class ResearchTask:
    question: str
    max_sources: int = 10

    def validate(self) -> None:
        if not self.question.strip():
            raise ValueError("question is required")
        if not 1 <= self.max_sources <= 100:
            raise ValueError("max_sources must be between 1 and 100")


@dataclass(frozen=True)
class ResearchSource:
    source_id: str          # e.g. "wikipedia:Eiffel Tower"
    title: str
    text: str

    def validate(self) -> None:
        if not self.source_id or not self.text.strip():
            raise ValueError("source_id and text are required")


@dataclass(frozen=True)
class ResearchClaim:
    """One extracted sentence + the source it came from + support score."""
    claim: str
    source_id: str
    support: float          # lexical overlap [0,1] with the source


@dataclass(frozen=True)
class ResearchResult:
    task: ResearchTask
    claims: tuple[ResearchClaim, ...] = ()
    sources_used: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "question": self.task.question,
            "claims": [{"claim": c.claim, "source": c.source_id,
                        "support": round(c.support, 3)} for c in self.claims],
            "sources": list(self.sources_used),
            "limitations": list(self.limitations),
        }


# ---------------------------------------------------------------- extraction
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"\w{3,}", re.UNICODE)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.split(text) if len(s.strip()) > 25]


def _tokens(s: str) -> set[str]:
    return set(_WORD_RE.findall(s.lower()))


def jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    union = ta | tb
    return len(ta & tb) / len(union) if union else 0.0


def make_research_result(task: ResearchTask,
                         sources: list[ResearchSource]) -> ResearchResult:
    """Deterministically extract claims tied to their sources.

    A sentence becomes a claim only if its lexical support with its own
    source exceeds a floor - the tiny-model equivalent of 'grounded'.
    """
    task.validate()
    limitations: list[str] = []
    if not sources:
        return ResearchResult(
            task=task,
            limitations=("no evidence supplied - refusing to fabricate",))

    claims: list[ResearchClaim] = []
    q_tokens = _tokens(task.question)
    seen: set[str] = set()
    for src in sources[:task.max_sources]:
        src.validate()
        for sent in _sentences(src.text):
            key = sent.lower()
            if key in seen:
                continue
            sup = jaccard(sent, src.text[:2000])
            relevance = jaccard(sent, task.question) \
                if q_tokens else 0.5
            # keep sentences either relevant to the question or strongly
            # representative of the source
            if sup >= 0.55 and (relevance > 0 or sup >= 0.7):
                seen.add(key)
                claims.append(ResearchClaim(claim=sent,
                                            source_id=src.source_id,
                                            support=sup))
    if not claims:
        limitations.append("no sentence reached the grounding threshold")
    if len(sources) < 2:
        limitations.append("single-source result - low corroboration")
    return ResearchResult(
        task=task,
        claims=tuple(claims),
        sources_used=tuple(s.source_id for s in sources[:task.max_sources]),
        limitations=tuple(limitations))
