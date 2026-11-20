"""The production-shaped Policy Assistant: filters, hybrid search, rerank, cited answer."""
import time

from . import db
from .documents import load_all
from .prompt import GUARD_RULE, NOT_FOUND, SYSTEM, build_messages, check_citations
from .providers import OllamaChat, OllamaEmbedder
from .rerank import CrossEncoderReranker
from .search import hybrid_search


class PolicyAssistant:
    def __init__(self, embedder=None, llm=None, reranker=None, conn=None, min_score=-4.0):
        self.conn = conn or db.connect()
        self.embedder = embedder or OllamaEmbedder()
        self.llm = llm or OllamaChat()
        self.reranker = reranker or CrossEncoderReranker()
        self.docs = {d.id: d for d in load_all()}
        self.min_score = min_score            # below this top reranker score, refuse without the LLM

    def retrieve(self, question, groups, k=4, pool=20):
        candidates = hybrid_search(self.conn, self.embedder, question, k=pool, pool=pool,
                                   groups=groups, current_only=True)
        return self.reranker.rerank(question, candidates, top=k)

    def ask(self, question, groups=("employees",)):
        timings = {}
        t = time.perf_counter()
        hits = self.retrieve(question, list(groups))
        timings["retrieve_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        if not hits or hits[0].score < self.min_score:
            answer, usage, sources = NOT_FOUND, {"prompt_tokens": 0, "output_tokens": 0}, []
        else:
            messages, sources = build_messages(question, hits, self.docs, system=SYSTEM + GUARD_RULE)
            answer, usage = self.llm.complete(messages)
        timings["llm_ms"] = (time.perf_counter() - t) * 1000
        return {"question": question, "answer": answer, "sources": sources,
                "check": check_citations(answer, sources), "usage": usage, "timings": timings}
