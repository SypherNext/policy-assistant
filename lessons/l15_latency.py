"""Where the time goes in one answer, and what caching saves."""
import hashlib
import statistics
import time

from policy_assistant import db
from policy_assistant.documents import load_all
from policy_assistant.prompt import build_messages
from policy_assistant.providers import OllamaChat, OllamaEmbedder
from policy_assistant.rerank import CrossEncoderReranker
from policy_assistant.search import keyword_search, or_query, rrf, vector_search

conn = db.connect()
embedder, llm, reranker = OllamaEmbedder(), OllamaChat(), CrossEncoderReranker()
docs = {d.id: d for d in load_all()}
questions = ["What is the hotel limit in Paris?", "How many characters must my password have?",
             "Can I work from Portugal for a month?", "What is form HR-12?",
             "How much do I get per kilometre for driving my own car?"]

stages = {k: [] for k in ("embed question", "vector search", "keyword search", "rerank 20",
                          "LLM read prompt", "LLM write answer")}
tokens_in, tokens_out = [], []


def timed(name, fn):
    t = time.perf_counter()
    out = fn()
    stages[name].append((time.perf_counter() - t) * 1000)
    return out


llm.complete([{"role": "user", "content": "Say OK."}], max_tokens=2)   # load the model first
for q in questions:
    qv = timed("embed question", lambda: embedder.embed_query(q))
    vec = timed("vector search", lambda: vector_search(conn, qv, 20, ["employees"], True))
    kw = timed("keyword search", lambda: keyword_search(conn, or_query(q), 20, ["employees"], True))
    best = timed("rerank 20", lambda: reranker.rerank(q, rrf([vec, kw], top=20), top=4))
    messages, _ = build_messages(q, best, docs)
    _, usage = llm.complete(messages)
    stages["LLM read prompt"].append(usage["prompt_ms"])
    stages["LLM write answer"].append(usage["output_ms"])
    tokens_in.append(usage["prompt_tokens"])
    tokens_out.append(usage["output_tokens"])

print(f"{len(questions)} questions, median per stage:")
for name, values in stages.items():
    print(f"  {name:18s} {statistics.median(values):9.1f} ms")
print(f"  prompt tokens median {statistics.median(tokens_in):.0f}, output tokens median {statistics.median(tokens_out):.0f}")


class AnswerCache:
    """Key = normalised question + who asks + index version, so an edit or a role change never reuses a stale answer."""

    def __init__(self):
        self.store = {}

    def key(self, question, groups, index_version):
        norm = " ".join(question.lower().split()).strip("?!. ")
        return hashlib.sha256(f"{norm}|{sorted(groups)}|{index_version}".encode()).hexdigest()

    def get_or_compute(self, question, groups, index_version, compute):
        k = self.key(question, groups, index_version)
        if k not in self.store:
            self.store[k] = compute()
        return self.store[k]


def full_answer(q):
    qv = embedder.embed_query(q)
    pool = rrf([vector_search(conn, qv, 20, ["employees"], True),
                keyword_search(conn, or_query(q), 20, ["employees"], True)], top=20)
    return llm.complete(build_messages(q, reranker.rerank(q, pool, top=4), docs)[0])[0]


index_version = conn.execute(
    "SELECT md5(string_agg(id || ':' || content_hash, ',' ORDER BY id)) FROM documents").fetchone()[0]
cache = AnswerCache()
print(f"\nindex version {index_version[:12]}")
for q in ["What is the hotel limit in Paris?", "what is the hotel limit in paris", "What is the hotel limit in Paris?"]:
    t = time.perf_counter()
    a = cache.get_or_compute(q, ["employees"], index_version, lambda: full_answer(q))
    print(f"  {(time.perf_counter() - t) * 1000:9.1f} ms  {q!r} -> {a}")
