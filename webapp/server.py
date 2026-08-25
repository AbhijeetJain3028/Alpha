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
from indus.model import IndusLM                   # noqa: E402
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

    def tokenizer(self):
        if self.tok is None:
            for p in ("data_v2/tokenizer.json", "data/tokenizer.json"):
                if os.path.exists(p):
                    self.tok = BPETokenizer.load(p)
                    break
            assert self.tok, "tokenizer.json not found"
        return self.tok

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
            self.model, self.cfg = model, cfg
            self.kind = ckpt.get("kind", "pretrained")
        return True


MM = ModelManager()


class ChatRequest(BaseModel):
    session_id: str
    message: str
    temperature: float = 0.7
    top_k: int = 50
    max_tokens: int = 256


SESSIONS: dict[str, dict] = {}


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
        "checkpoint": MM.ckpt_name,
        "kind": MM.kind,
        "vocab_size": MM.cfg.vocab_size,
        "block_size": MM.cfg.block_size,
        "device": MM.device,
        "chat_ready": "<|assistant|>" in MM.tokenizer().special_tokens
                      and MM.cfg.vocab_size >= len(MM.tokenizer().vocab),
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

    def gen():
        with MM.lock:
            tok = MM.tokenizer()
            eot = tok.special_tokens.get("<|endoftext|>")
            end = tok.special_tokens.get("<|end|>")
            specials = set(tok.special_tokens.values())
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--which", default="ckpt-latest.pt")
    args = ap.parse_args()
    os.environ["INDUS_WHICH"] = args.which
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
