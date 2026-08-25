#!/usr/bin/env python3
"""Indus LLM <-> Indus-Kernel bridge: a stdio JSON-RPC tool server.

This is the integration path that lets ALL 40 kernel subsystems (or any
MCP-style agent stack) drive the model: the kernel's `ik_tools` can list
and call these tools; `ik_agents` orchestrates them; `ik_eval` audits the
trace. Protocol: newline-delimited JSON-RPC 2.0 over stdin/stdout - zero
dependencies, kernel-agnostic.

Methods:
  initialize           -> capability handshake
  tools/list           -> available tools + JSON schemas
  tools/call           -> {name, arguments}
     chat       {message, temperature?, max_tokens?}
     research   {topic, train?, steps?}   # full autonomy cycle
     answer     {question, k?}            # grounded from knowledge store
     info       {}
Example:
  echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' | python scripts/indus_mcp_server.py
"""

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indus.autonomous import (KnowledgeStore, SelfLearner, WebCorpus,    # noqa: E402
                              answer_grounded, autonomous_cycle)
from indus.config import IndusConfig                                    # noqa: E402
from indus.constitution import enforce as const_enforce                 # noqa: E402
from indus.model import IndusLM, ensure_vocab_size                      # noqa: E402
from indus.research_contract import ResearchSource, make_research_result  # noqa: E402
from indus.tokenizer import BPETokenizer                                # noqa: E402

CKPT_CHAIN = ["checkpoints/ckpt-sft.pt", "checkpoints/ckpt-online.pt",
              "checkpoints/ckpt-final.pt", "checkpoints/ckpt-latest.pt"]

TOOLS = [
    {"name": "chat", "description": "Talk to Indus",
     "input_schema": {"type": "object", "properties": {
         "message": {"type": "string"},
         "temperature": {"type": "number", "default": 0.7},
         "max_tokens": {"type": "integer", "default": 80}}}},
    {"name": "research", "description":
        "Autonomous cycle: web research -> knowledge store -> eval-gated "
        "self-training -> grounded answer with auditable claims",
     "input_schema": {"type": "object", "properties": {
         "topic": {"type": "string"},
         "train": {"type": "boolean", "default": True},
         "steps": {"type": "integer", "default": 40}}}},
    {"name": "answer", "description":
        "Answer a question from the existing knowledge store",
     "input_schema": {"type": "object", "properties": {
         "question": {"type": "string"}, "k": {"type": "integer"}}}},
    {"name": "info", "description": "Model + store capabilities",
     "input_schema": {"type": "object", "properties": {}}},
]


class IndusService:
    def __init__(self, ckpt: str | None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        path = ckpt or next(
            (c for c in CKPT_CHAIN if os.path.exists(c)), None)
        if path is None:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download("AbhijeetJain4075/indus-llm",
                                   "ckpt-latest.pt", repo_type="model")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = IndusConfig.from_dict(ck["config"])
        self.model = IndusLM(cfg).to(self.device)
        self.model.load_state_dict(
            {k: v.cpu() for k, v in ck["model"].items()}, strict=False)
        for cand in ("data/tokenizer.json", "data_v2/tokenizer.json"):
            if os.path.exists(cand):
                self.tok = BPETokenizer.load(cand)
                break
        else:
            raise SystemExit("tokenizer.json not found")
        self.tok.add_chat_specials()
        ensure_vocab_size(self.model, len(self.tok.vocab))
        self.model.eval()
        self.kind = ck.get("kind", "pretrained")
        self.store = KnowledgeStore()
        replay = "data/train.bin" if os.path.exists("data/train.bin") else None
        self.learner = SelfLearner(self.model, self.tok,
                                   device=self.device, replay_bin=replay)
        self.corpus = WebCorpus()

    def _generate(self, prompt: str, temperature: float = 0.7,
                  max_new_tokens: int = 80) -> str:
        ids = self.tok.encode_with_specials(prompt)[-self.model.config.block_size:]
        x = torch.tensor([ids], device=self.device)
        eot = self.tok.special_tokens.get("<|endoftext|>")
        end = self.tok.special_tokens.get("<|end|>")
        y = self.model.generate(x, max_new_tokens=max_new_tokens,
                                temperature=temperature, top_k=50,
                                endoftext_id=eot)
        out = y[0].tolist()[len(ids):]
        if end is not None and end in out:
            out = out[:out.index(end)]
        return self.tok.decode(out).strip()

    def chat(self, message: str, temperature: float = 0.7,
             max_tokens: int = 80) -> dict:
        hits = self.store.search(message, k=2) if self.store.count() else []
        prompt = (f"<|user|>\n{message}<|end|>\n<|assistant|>\n"
                  if not hits else
                  __import__("indus.autonomous", fromlist=["grounded_prompt"])
                  .grounded_prompt(self.tok, message, hits))
        reply, verdict = const_enforce(
            lambda p, **kw: self._generate(p, **kw), prompt,
            temperature=temperature, max_new_tokens=max_tokens)
        return {"reply": reply, "grounded": bool(hits),
                "constitution": verdict}

    def research(self, topic: str, train: bool = True,
                 steps: int = 40) -> dict:
        rep = autonomous_cycle(self.model, self.tok, topic,
                               corpus=self.corpus, store=self.store,
                               learner=self.learner if train else None,
                               device=self.device, verbose=False)
        srcs = [ResearchSource(source_id=d["source"], title=d["title"],
                               text=d["text"])
                for d in self.corpus.gather(topic)]
        from indus.research_contract import ResearchTask
        rr = make_research_result(ResearchTask(question=f"What is {topic}?"),
                                  srcs)
        rep["auditable_claims"] = rr.to_dict()
        return rep

    def answer(self, question: str, k: int = 3) -> dict:
        hits = self.store.search(question, k=k)
        if not hits:
            return {"reply": "", "citations": [],
                    "limitations": ["knowledge store empty for this query"]}
        reply, cites = answer_grounded(self.model, self.tok, question,
                                       hits, device=self.device)
        return {"reply": reply, "citations": cites}

    def info(self) -> dict:
        return {"model": self.model.config.name,
                "kind": self.kind,
                "params_M": round(self.model.num_params() / 1e6, 2),
                "vocab": len(self.tok.vocab),
                "knowledge_docs": self.store.count(),
                "device": self.device}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()

    svc = IndusService(args.ckpt)

    def rpc(resp: dict) -> None:
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            rpc({"jsonrpc": "2.0", "id": None,
                 "error": {"code": -32700, "message": "parse error"}})
            continue
        rid, method = req.get("id"), req.get("method")
        params = req.get("params") or {}
        try:
            if method == "initialize":
                res = {"protocolVersion": "2026-07-28",
                       "serverInfo": {"name": "indus-llm",
                                      "version": "0.1.0"}}
            elif method == "tools/list":
                res = {"tools": TOOLS}
            elif method == "tools/call":
                name = params.get("name")
                a = params.get("arguments") or {}
                if name == "chat":
                    out = svc.chat(**a)
                elif name == "research":
                    out = svc.research(**a)
                elif name == "answer":
                    out = svc.answer(**a)
                elif name == "info":
                    out = svc.info()
                else:
                    raise ValueError(f"unknown tool: {name}")
                res = {"content": [{"type": "text",
                                    "text": json.dumps(out)}]}
            else:
                rpc({"jsonrpc": "2.0", "id": rid,
                     "error": {"code": -32601,
                               "message": f"unknown method {method}"}})
                continue
            rpc({"jsonrpc": "2.0", "id": rid, "result": res})
        except Exception as e:
            rpc({"jsonrpc": "2.0", "id": rid,
                 "error": {"code": -32000, "message": str(e)}})


if __name__ == "__main__":
    main()
