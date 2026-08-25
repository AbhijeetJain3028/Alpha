#!/usr/bin/env python3
"""Indus web server - a claude.ai-style interface for the Indus LLM.

Serves the branded single-page frontend and streams generations over SSE.

  python webapp/server.py --which ckpt-sft.pt --port 8000
"""

import argparse
import json
import os
import sys
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from indus.config import IndusConfig              # noqa: E402
from indus.autonomous import (KnowledgeStore, SelfLearner, WebCorpus,  # noqa: E402
                              answer_grounded, grounded_prompt)
from indus.model import IndusLM, ensure_vocab_size  # noqa: E402
from indus.tokenizer import BPETokenizer          # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

app = FastAPI(title="Indus", docs_url=None, redoc_url=None)


class ModelManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.tok = None
        self.model = None
        self.cfg = None
        self.ckpt_name = "?"
        self.kind = "unloaded"
        self.device = "cpu"
        self.corpus: WebCorpus | None = None
        self.store: KnowledgeStore | None = None
        self.learner: SelfLearner | None = None

    def tokenizer(self):
        if self.tok is None:
            for p in ("data_v2/tokenizer.json", "data/tokenizer.json"):
                if os.path.exists(p):
                    self.tok = BPETokenizer.load(p)
                    break
            assert self.tok, "tokenizer.json not found"
        self.tok.add_chat_specials()   # idempotent
        return self.tok

    def autonomy(self):
        """Lazily create the research corpus + knowledge store + learner."""
        if self.corpus is None:
            self.corpus = WebCorpus()
        if self.store is None:
            self.store = KnowledgeStore(os.path.join(
                ROOT, "knowledge", "knowledge.db"))
        if self.learner is None and self.model is not None:
            replay = os.path.join(ROOT, "data", "train.bin")
            self.learner = SelfLearner(
                self.model, self.tokenizer(), device=self.device,
                replay_bin=replay if os.path.exists(replay) else None)
        return self.corpus, self.store, self.learner

    def load(self, ckpt_path=None, which="ckpt-latest.pt", device=None):
        if ckpt_path is None:
            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(
                os.environ.get("HF_REPO", "AbhijeetJain4075/indus-llm"),
                which, repo_type="model")
            self.ckpt_name = which
        else:
            self.ckpt_name = os.path.basename(ckpt_path)
        tok = self.tokenizer()
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = IndusConfig.from_dict(ckpt["config"])
        if len(tok.vocab) != cfg.vocab_size:
            cfg.vocab_size = max(cfg.vocab_size, len(tok.vocab))
        with self.lock:
            self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            model = IndusLM(cfg).to(self.device).eval()
            state = {k: v.cpu() for k, v in ckpt["model"].items()}
            final = {k: v for k, v in state.items()
                     if k in model.state_dict()
                     and model.state_dict()[k].shape == v.shape}
            model.load_state_dict(final, strict=False)
            ensure_vocab_size(model, max(cfg.vocab_size,
                                         model.tok_emb.weight.shape[0]))
            self.model, self.cfg = model, cfg
            self.kind = ckpt.get("kind", "pretrained")
            self.learner = None          # rebind learner to fresh weights
        return True


MM = ModelManager()


class ChatRequest(BaseModel):
    session_id: str
    message: str
    temperature: float = 0.7
    top_k: int = 50
    max_tokens: int = 256
    system: str | None = None
    research: bool = False        # retrieval-grounded answering


class ResearchRequest(BaseModel):
    session_id: str
    topic: str
    train: bool = True            # also fine-tune on what was found
    steps: int = 60
    max_pages: int = 3


SESSIONS: dict[str, dict] = {}

FEEDBACK_PATH = os.path.join(ROOT, "data_feedback", "feedback.jsonl")


def log_feedback(entry: dict) -> None:
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), **entry}) + "\n")


def render_history(sess) -> str:
    return sess["history"]


def build_prompt(sess, message: str) -> list[int]:
    tok = MM.tokenizer()
    chat = "<|assistant|>" in tok.special_tokens
    sys_p = sess.get("system", "")
    if chat:
        prefix = f"<|system|>\n{sys_p}<|end|>\n" if sys_p and not sess["turns"] else ""
        ids = tok.encode_with_specials(
            prefix + sess["history"] +
            f"<|user|>\n{message}<|end|>\n<|assistant|>\n")
    else:
        ids = tok.encode(sess["history"] + message + "\n")
    return ids[-MM.cfg.block_size:]


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/static/{path}")
def static_files(path: str):
    fp = os.path.join(STATIC, path)
    if not os.path.exists(fp):
        raise HTTPException(404)
    return FileResponse(fp)


@app.get("/api/info")
def info():
    if MM.model is None:
        try:
            MM.load(which=os.environ.get("INDUS_WHICH", "ckpt-latest.pt"))
        except Exception as e:
            raise HTTPException(503, f"model not loaded: {e}")
    return {
        "name": MM.cfg.name,
        "params_M": round(MM.model.num_params() / 1e6, 1),
        "params_total_M":
            round(MM.model.num_params(non_embedding=False) / 1e6, 2),
        "checkpoint": MM.ckpt_name,
        "kind": MM.kind,
        "vocab_size": MM.cfg.vocab_size,
        "block_size": MM.cfg.block_size,
        "device": MM.device,
        "chat_ready": "<|assistant|>" in MM.tokenizer().special_tokens
                      and MM.cfg.vocab_size >= len(MM.tokenizer().vocab),
        "knowledge_docs": (MM.store.count() if MM.store else None),
    }


class LoadReq(BaseModel):
    which: str | None = None
    path: str | None = None


@app.post("/api/load")
def load_model(req: LoadReq):
    try:
        MM.load(ckpt_path=req.path, which=req.which or "ckpt-latest.pt")
        return {"ok": True, "checkpoint": MM.ckpt_name}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/api/chat")
def chat(req: ChatRequest):
    if MM.model is None:
        info()  # triggers lazy load; raises 503 on failure
    sess = SESSIONS.setdefault(req.session_id,
                               {"history": "", "turns": 0, "system": ""})
    if req.system is not None and req.system.strip():
        if req.system != sess.get("system"):
            sess["system"] = req.system.strip()
            sess["history"], sess["turns"] = "", 0   # system change resets ctx

    def gen():
        with MM.lock:
            tok = MM.tokenizer()
            eot = tok.special_tokens.get("<|endoftext|>")
            end = tok.special_tokens.get("<|end|>")
            specials = set(tok.special_tokens.values())
            if req.research:
                _, store, _ = MM.autonomy()
                hits = store.search(req.message, k=3) \
                    if store.count() else []
                ids = tok.encode_with_specials(
                    grounded_prompt(tok, req.message, hits))[
                        -MM.cfg.block_size:] if hits else \
                    build_prompt(sess, req.message)
            else:
                ids = build_prompt(sess, req.message)
            x = torch.tensor([ids], device=MM.device)
            reply_chunks = []
            t0 = time.time()
            n_new = 0
            yield sse({"t": "start"})
            for _ in range(min(req.max_tokens, MM.cfg.block_size)):
                logits = MM.model(x[:, -MM.cfg.block_size:]).logits[0, -1]
                if req.temperature <= 1e-6:
                    nxt = int(torch.argmax(logits))
                else:
                    lg = logits / req.temperature
                    if req.top_k:
                        k = min(req.top_k, lg.size(-1))
                        lg = lg.masked_fill(
                            lg < torch.topk(lg, k).values[-1], float("-inf"))
                    nxt = int(torch.multinomial(torch.softmax(lg, -1), 1))
                if nxt in (eot, end):
                    break
                n_new += 1
                if nxt not in specials:
                    piece = tok.vocab[nxt].decode("utf-8", errors="replace")
                    reply_chunks.append(piece)
                    yield sse({"t": "tok", "v": piece})
                x = torch.cat([x, torch.tensor([[nxt]], device=MM.device)],
                              dim=1)
            rate = n_new / max(time.time() - t0, 1e-6)
            reply = "".join(reply_chunks).strip()
            chat_mode = "<|assistant|>" in tok.special_tokens
            sess["history"] += (
                f"<|user|>\n{req.message}<|end|>\n<|assistant|>\n{reply}<|end|>\n"
                if chat_mode else sess["history"] + req.message + "\n")
            sess["turns"] += 1
            if len(sess["history"]) > MM.cfg.block_size * 6:
                sess["history"] = sess["history"][len(sess["history"]) // 2:]
            yield sse({"t": "done", "rate": round(rate, 1),
                       "tokens": n_new})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


def sse(obj) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.post("/api/research")
def research(req: ResearchRequest):
    """Full autonomy cycle: web research -> knowledge store -> eval-gated
    self-training -> grounded answer, streamed as SSE."""
    if MM.model is None:
        info()
    sess = SESSIONS.setdefault(req.session_id,
                               {"history": "", "turns": 0, "system": ""})

    def gen():
        with MM.lock:
            tok = MM.tokenizer()
            corpus, store, learner = MM.autonomy()
            yield sse({"t": "status", "v": f"researching '{req.topic}' ..."})
            docs = []
            try:
                corpus.max_pages = req.max_pages
                docs = corpus.gather(req.topic)
            except Exception as e:
                yield sse({"t": "status", "v": f"research error: {e}"})
            for d in docs:
                store.add(d)
            titles = [f"{d['title']} ({d['source'].split(':')[0]})"
                      for d in docs]
            yield sse({"t": "sources", "v": titles})
            if not docs:
                yield sse({"t": "done", "rate": 0, "tokens": 0})
                return

            if req.train and learner is not None:
                yield sse({"t": "status",
                           "v": f"learning from {len(docs)} sources "
                                f"({req.steps} steps, eval-gated) ..."})
                try:
                    res = learner.learn(docs, steps=req.steps)
                    yield sse({"t": "learn", "v": json.dumps(res)})
                except Exception as e:
                    yield sse({"t": "status", "v": f"learn error: {e}"})

            hits = store.search(req.topic, k=3)
            question = f"What is {req.topic}?"
            prompt_ids = tok.encode_with_specials(
                grounded_prompt(tok, question, hits))[-MM.cfg.block_size:]
            x = torch.tensor([prompt_ids], device=MM.device)
            eot = tok.special_tokens.get("<|endoftext|>")
            end = tok.special_tokens.get("<|end|>")
            specials = set(tok.special_tokens.values())
            reply_chunks = []
            n_new = 0
            t0 = time.time()
            yield sse({"t": "status", "v": "answering with sources ..."})
            yield sse({"t": "tok", "v": ""})     # flush
            for _ in range(min(110, MM.cfg.block_size)):
                logits = MM.model(x[:, -MM.cfg.block_size:]).logits[0, -1]
                lg = logits / max(0.5, 1e-6)
                k = min(50, lg.size(-1))
                lg = lg.masked_fill(
                    lg < torch.topk(lg, k).values[-1], float("-inf"))
                nxt = int(torch.multinomial(torch.softmax(lg, -1), 1))
                if nxt in (eot, end):
                    break
                n_new += 1
                if nxt not in specials:
                    piece = tok.vocab[nxt].decode("utf-8", errors="replace")
                    reply_chunks.append(piece)
                    yield sse({"t": "tok", "v": piece})
                x = torch.cat([x, torch.tensor([[nxt]], device=MM.device)],
                              dim=1)
            reply = "".join(reply_chunks).strip()
            rate = n_new / max(time.time() - t0, 1e-6)
            sess["history"] += (f"<|user|>\n{question}<|end|>\n"
                                f"<|assistant|>\n{reply}<|end|>\n")
            sess["turns"] += 1
            yield sse({"t": "done", "rate": round(rate, 1),
                       "tokens": n_new})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


class FeedbackReq(BaseModel):
    session_id: str
    prompt: str
    reply: str
    verdict: str            # "up" | "down" | "taught"
    correction: str | None = None


@app.post("/api/feedback")
def feedback(req: FeedbackReq):
    if req.verdict not in ("up", "down", "taught"):
        raise HTTPException(400, "verdict must be up|down|taught")
    log_feedback({
        "session": req.session_id,
        "prompt": req.prompt[:2000],
        "reply": req.reply[:4000],
        "verdict": req.verdict,
        "correction": (req.correction or "")[:4000] or None,
    })
    return {"ok": True}


@app.get("/api/feedback/stats")
def feedback_stats():
    counts = {"up": 0, "down": 0, "taught": 0}
    if os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH, encoding="utf-8") as f:
            for line in f:
                try:
                    counts[json.loads(line).get("verdict")] += 1
                except Exception:
                    continue
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--which", default="ckpt-latest.pt")
    args = ap.parse_args()
    os.environ["INDUS_WHICH"] = args.which
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
