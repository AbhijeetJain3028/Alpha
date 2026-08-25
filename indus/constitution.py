"""Generation-time constitutional scaffold - ported from Indus-Kernel
`ik_indus_llm/constitution.py`, adapted for a tiny model.

Peak-logic adaptation: an 11M model cannot reliably self-critique in text,
so the constitution is enforced by the SCAFFOLD, not by the network:
  1. principles are injected as a system prompt prefix;
  2. deterministic post-generation linting flags violations
     (PII patterns, harmful-instruction triggers) and can trigger a
     single temperature-lowered regenerate.
Reference: Bai et al., 2022 - Constitutional AI (arXiv:2212.08073).
"""

from __future__ import annotations

import re

DEFAULT_CONSTITUTION = [
    "Be helpful, accurate, and concise.",
    "No hateful, harassing, or violent content.",
    "No instructions for weapons, malware, or other harmful tools.",
    "When uncertain, say so rather than fabricate.",
    "Respect privacy; never output personal data.",
]

_PII = [
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),                 # email
    re.compile(r"\b(?:\d[ -]?){13,16}\b"),                      # card-like
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),                       # ssn-like
]
_HARM = re.compile(
    r"\b(build|make|create)\s+(a\s+)?(bomb|virus|malware|exploit)\b", re.I)


def system_prefix(principles: list[str] | None = None) -> str:
    p = principles or DEFAULT_CONSTITUTION
    return "Follow these rules strictly:\n" + \
        "\n".join(f"- {r}" for r in p)


def lint(reply: str) -> dict:
    """Deterministic post-generation check. Returns verdict dict."""
    violations = []
    if any(p.search(reply) for p in _PII):
        violations.append("pii")
    if _HARM.search(reply):
        violations.append("harmful_instructions")
    return {"clean": not violations, "violations": violations}


def enforce(generate_fn, prompt: str, max_attempts: int = 2, **gen_kwargs):
    """Call generate_fn(prompt); if lint fails, retry once at lower temp."""
    reply = generate_fn(prompt, **gen_kwargs)
    for _ in range(max_attempts - 1):
        v = lint(reply)
        if v["clean"]:
            break
        gen_kwargs["temperature"] = min(gen_kwargs.get("temperature", 0.8),
                                        0.3)
        reply = generate_fn(prompt, **gen_kwargs)
    return reply, lint(reply)
