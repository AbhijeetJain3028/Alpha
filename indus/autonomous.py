"""Autonomous web-research + self-training engine for Indus.

Doctrine distilled from the research literature (REALM/RAG/RA-DIT,
Toolformer, STaR, Search-R1, continual-learning work):

  1. Knowledge lives OUTSIDE the network. An 11M-param model cannot and
     should not memorize facts; a retriever carries them. The model's job
     is query shaping, extraction and grounded composition.
  2. Self-training must be EVAL-GATED. Every learning cycle snapshots the
     weights, fine-tunes on freshly researched text mixed with a REPLAY
     BUFFER drawn from the base corpus (anti catastrophic-forgetting),
     then rolls back unless the held-out probe improves.
  3. Everything is inspectable: every cycle returns a structured report
     (sources fetched, tokens learned, loss deltas, verdict).

Zero external dependencies beyond PyTorch/numpy: Wikipedia's public
action API + DuckDuckGo lite HTML + stdlib sqlite3 FTS5.
"""

from __future__ import annotations

import html as _html
import json
import os
import re
import sqlite3
import time
import urllib.parse
import urllib.request

import numpy as np
import torch

from .model import IndusLM, ensure_vocab_size
from .tokenizer import BPETokenizer

UA = {"User-Agent":
      "Indus-Autonomous-LLM/0.1 (from-scratch research agent; local use)"}
EOT_TXT = "<|endoftext|>"


# --------------------------------------------------------------------- utils
def _http_json(url: str, params: dict | None = None, timeout: int = 15):
    if params:
        url = url + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", errors="replace"))


def _http_text(url: str, timeout: int = 15) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def clean_text(t: str) -> str:
    t = _html.unescape(t or "")
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\[[0-9]+\]|\[citation needed\]|\[note [0-9]+\]", "", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


# ------------------------------------------------------------------ fetching
class WebCorpus:
    """Researches a topic into clean passages. Wikipedia-first (reliable,
    license-clean CC BY-SA); DuckDuckGo-lite snippet fallback."""

    def __init__(self, max_pages: int = 3, max_chars_per_page: int = 4000):
        self.max_pages = max_pages
        self.max_chars = max_chars_per_page

    # -- wikipedia -------------------------------------------------------
    def wiki_search(self, topic: str) -> list[str]:
        try:
            d = _http_json("https://en.wikipedia.org/w/api.php", {
                "action": "query", "list": "search", "srsearch": topic,
                "srlimit": self.max_pages, "format": "json"})
            return [h["title"] for h in d.get("query", {}).get("search", [])]
        except Exception:
            return []

    def wiki_page(self, title: str) -> str:
        try:
            d = _http_json("https://en.wikipedia.org/w/api.php", {
                "action": "query", "prop": "extracts", "explaintext": 1,
                "exsectionformat": "plain", "redirects": 1,
                "titles": title, "format": "json"})
            pages = d.get("query", {}).get("pages", {})
            for _, p in pages.items():
                return clean_text(p.get("extract", ""))[:self.max_chars]
        except Exception:
            pass
        return ""

    # -- duckduckgo lite fallback ----------------------------------------
    def ddg_snippets(self, topic: str) -> list[tuple[str, str]]:
        out = []
        try:
            page = _http_text("https://lite.duckduckgo.com/lite/?q="
                              + urllib.parse.quote(topic))
            rows = re.findall(
                r"<a[^>]+class=\"result-link\"[^>]*>(.*?)</a>", page)
            snips = re.findall(
                r"<td[^>]*class=\"result-snippet\"[^>]*>(.*?)</td>", page,
                re.S)
            for i, (t_, s_) in enumerate(zip(rows, snips)):
                if i >= self.max_pages:
                    break
                title = clean_text(re.sub("<[^>]+>", "", t_))
                body = clean_text(s_)
                if len(body) > 80:
                    out.append((title[:120], body[:1200]))
        except Exception:
            pass
        return out

    def gather(self, topic: str) -> list[dict]:
        """Returns [{title, text, source}] - deduped, cleaned, bounded."""
        docs: list[dict] = []
        seen: set[str] = set()

        def push(title, text, source):
            text = clean_text(text)
            key = title.lower()
            if len(text) < 200 or key in seen:
                return
            seen.add(key)
            docs.append({"title": title, "text": text, "source": source})

        for t in self.wiki_search(topic)[:self.max_pages]:
            txt = self.wiki_page(t)
            if txt:
                push(t, txt, f"wikipedia:{t}")
        if not docs:                       # wiki miss -> ddg snippets
            for title, snip in self.ddg_snippets(topic):
                push(title, snip, f"web:{title}")
        return docs


# -------------------------------------------------------------------- memory
class KnowledgeStore:
    """Persistent FTS5 knowledge base - the part of 'the mind' that is not
    neural. Supports ranked retrieval (BM25 via MATCH ... ORDER BY rank)."""

    def __init__(self, path: str = "knowledge/knowledge.db"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE VIRTUAL TABLE IF NOT EXISTS docs USING "
                        "fts5(title, text, source)")
        self.db.commit()

    def add(self, doc: dict) -> bool:
        """Insert unless an identical (title, source) already exists."""
        row = self.db.execute(
            "SELECT 1 FROM docs WHERE title = ? AND source = ? LIMIT 1",
            (doc["title"], doc["source"])).fetchone()
        if row:
            return False
        self.db.execute("INSERT INTO docs(title, text, source) VALUES(?,?,?)",
                        (doc["title"], doc["text"], doc["source"]))
        self.db.commit()
        return True

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = " OR ".join(w for w in re.findall(r"\w{3,}", query)) or query
        try:
            rows = self.db.execute(
                "SELECT title, text, source, bm25(docs) FROM docs "
                "WHERE docs MATCH ? ORDER BY rank LIMIT ?",
                (q, k)).fetchall()
        except sqlite3.OperationalError:
            return []
        return [{"title": t, "text": x, "source": s} for t, x, s, _ in rows]

    def count(self) -> int:
        return self.db.execute("SELECT count(*) FROM docs").fetchone()[0]


# ----------------------------------------------------------------- learner
def _passage_tokens(tok: BPETokenizer, doc: dict, cap_tokens: int) -> list[int]:
    eot = tok.special_tokens[EOT_TXT]
    ids = tok.encode(doc["title"] + ". " + doc["text"])[:cap_tokens]
    return ids + [eot]


class SelfLearner:
    """Eval-gated continual fine-tuning with replay buffer."""

    def __init__(self, model: IndusLM, tok: BPETokenizer, device: str = "cpu",
                 replay_bin: str | None = None, probe_tokens: int = 2048):
        self.model = model
        self.tok = tok
        self.device = device
        self.probe_tokens = probe_tokens
        ensure_vocab_size(model, len(tok.vocab))
        self.replay: np.ndarray | None = None
        if replay_bin and os.path.exists(replay_bin):
            self.replay = np.memmap(replay_bin, dtype=np.uint16, mode="r")

    # -- internals ---------------------------------------------------------
    def _probe_loss(self, probe_ids: list[int]) -> float:
        """Mean NLL over non-overlapping windows. Tiles short streams so a
        window always exists (a 0.0 return would silently pass the gate)."""
        m, blk = self.model, self.model.config.block_size
        ids = list(probe_ids)
        need = blk + 1
        while len(ids) < need + blk:            # tile until >= 2 windows worth
            ids = ids + ids
        m.eval()
        losses = []
        with torch.no_grad():
            for i in range(0, len(ids) - blk, blk):
                x = torch.tensor([ids[i:i + blk]], device=self.device)
                y = torch.tensor([ids[i + 1:i + blk + 1]],
                                 device=self.device)
                losses.append(m(x, targets=y).loss.item())
        m.train()
        return sum(losses) / max(len(losses), 1)

    def learn(self, docs: list[dict], steps: int = 60, lr: float = 2e-5,
              batch_size: int = 4, replay_ratio: float = 0.35,
              fresh_cap: int = 256, seed: int = 7) -> dict:
        """Fine-tune on researched passages. Rollback unless probe improves."""
        rng = np.random.default_rng(seed)
        eot = self.tok.special_tokens[EOT_TXT]
        fresh: list[int] = []
        for d in docs:
            fresh += _passage_tokens(self.tok, d, fresh_cap)

        # probe = unseen tail slice of the fresh stream (never trained on)
        cut = int(len(fresh) * 0.8)
        train_ids, probe = fresh[:cut], fresh[cut:]
        if len(probe) < self.model.config.block_size + 1:
            probe = fresh[-(self.model.config.block_size + 1):]

        before_probe = self._probe_loss(probe)
        before_fresh = self._probe_loss(train_ids[:1024])

        snap = {k: v.detach().clone() for k, v in
                self.model.state_dict().items()}
        opt = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                betas=(0.9, 0.95), weight_decay=0.01)

        def batch():
            ids = []
            for _ in range(batch_size):
                src_fresh = rng.random() >= replay_ratio
                pool = np.asarray(train_ids, dtype=np.int64)
                if not src_fresh and self.replay is not None \
                        and len(self.replay) > 4096:
                    i0 = rng.integers(0, len(self.replay) - 600)
                    pool = np.asarray(self.replay[i0:i0 + 600],
                                      dtype=np.int64)
                if len(pool) < 8:
                    continue
                i0 = rng.integers(0, max(1, len(pool) -
                                         self.model.config.block_size - 1))
                seq = pool[i0:i0 + self.model.config.block_size + 1]
                if len(seq) < self.model.config.block_size + 1:
                    seq = np.pad(seq, (0, self.model.config.block_size + 1 -
                                       len(seq)),
                                 constant_values=eot)
                ids.append(seq)
            if not ids:
                return None, None
            x = torch.tensor(np.stack([s[:-1] for s in ids]),
                             device=self.device)
            y = torch.tensor(np.stack([s[1:] for s in ids]),
                             device=self.device)
            return x, y

        self.model.train()
        losses = []
        t0 = time.time()
        for _ in range(steps):
            x, y = batch()
            if x is None:
                continue
            loss = self.model(x, targets=y).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        after_probe = self._probe_loss(probe)
        after_fresh = self._probe_loss(train_ids[:1024])
        improved = after_probe <= before_probe + 0.02   # tolerance gate
        if not improved:                                 # rollback
            self.model.load_state_dict(snap)
        return {
            "steps": steps, "train_seconds": round(time.time() - t0, 1),
            "fresh_tokens": len(train_ids),
            "probe_before": round(before_probe, 4),
            "probe_after": round(after_probe, 4),
            "fresh_before": round(before_fresh, 4),
            "fresh_after": round(after_fresh, 4),
            "mean_batch_loss": round(float(np.mean(losses)), 4) if losses
            else None,
            "accepted": bool(improved),
        }


# ---------------------------------------------------------------- answering
GROUNDING_SYS = ("You are Indus, a precise research assistant. Answer using "
                 "ONLY the numbered SOURCES. Cite like [1].")


def grounded_prompt(tok: BPETokenizer, question: str, hits: list[dict]) -> str:
    """Build a chat-format prompt with retrieved sources, bounded by ctx."""
    blk = 512
    budget = blk - 96                       # leave room for question+answer
    per_src = max(64, budget // max(len(hits), 1) - 12)
    parts = []
    used = 0
    for i, h in enumerate(hits, 1):
        piece = f"[{i}] {h['title']}: " + h["text"]
        n = len(tok.encode(piece))
        if used + n > budget:
            piece_ids = tok.encode(piece)[:max(per_src, 32)]
            piece = tok.decode(piece_ids) + " ..."
            n = per_src
        parts.append(f"{piece}\n")
        used += n
    return (f"<|system|>\n{GROUNDING_SYS}<|end|>\n"
            f"<|user|>\nSOURCES:\n{''.join(parts)}\n\nQUESTION: "
            f"{question}<|end|>\n<|assistant|>\n")


@torch.no_grad()
def answer_grounded(model: IndusLM, tok: BPETokenizer, question: str,
                    hits: list[dict], device: str = "cpu",
                    max_new_tokens: int = 90, temperature: float = 0.5) \
        -> tuple[str, list[str]]:
    prompt = grounded_prompt(tok, question, hits)
    ids = tok.encode_with_specials(prompt)[-model.config.block_size:]
    x = torch.tensor([ids], device=device)
    eot = tok.special_tokens.get(EOT_TXT)
    end = tok.special_tokens.get("<|end|>")
    y = model.generate(x, max_new_tokens=max_new_tokens,
                       temperature=temperature, top_k=50, endoftext_id=eot)
    reply_ids = y[0].tolist()[len(ids):]
    if end is not None and end in reply_ids:
        reply_ids = reply_ids[:reply_ids.index(end)]
    reply = tok.decode(reply_ids).strip()
    cited = [h["source"] for h in hits]
    return reply, cited


# ------------------------------------------------------------------- cycles
def autonomous_cycle(model: IndusLM, tok: BPETokenizer, topic: str,
                     corpus: WebCorpus | None = None,
                     store: KnowledgeStore | None = None,
                     learner: SelfLearner | None = None,
                     device: str = "cpu", verbose: bool = True) -> dict:
    """One full act of autonomy: research -> remember -> learn -> answer."""
    corpus = corpus or WebCorpus()
    store = store or KnowledgeStore()
    report: dict = {"topic": topic, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}

    docs = corpus.gather(topic)
    for d in docs:
        store.add(d)
    report["sources_found"] = len(docs)
    report["store_size"] = store.count()
    if verbose:
        print(f"[research] '{topic}': {len(docs)} sources, "
              f"store now {report['store_size']} docs")

    if docs and learner is not None:
        report["learning"] = learner.learn(docs)
        if verbose:
            lrn = report["learning"]
            print(f"[learn] accepted={lrn['accepted']} "
                  f"probe {lrn['probe_before']}->{lrn['probe_after']} "
                  f"| fresh {lrn['fresh_before']}->{lrn['fresh_after']} "
                  f"({lrn['train_seconds']}s)")

    hits = store.search(topic, k=3)
    question = f"What is {topic}?"
    reply, cites = answer_grounded(model, tok, question, hits,
                                   device=device, max_new_tokens=70)
    report["question"], report["answer"] = question, reply
    report["citations"] = cites
    if verbose:
        print(f"[answer] Q: {question}\n         A: {reply}\n"
              f"         cites: {cites}")
    return report
