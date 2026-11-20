"""Retrieval: vector search, keyword search, and reciprocal rank fusion."""
from dataclasses import dataclass

import numpy as np


@dataclass
class Hit:
    chunk_id: int
    doc_id: str
    heading: str
    content: str
    score: float


def _where(groups, current_only):
    clauses, params = [], []
    if groups is not None:
        clauses.append("allowed_groups && %s::text[]")
        params.append(list(groups))
    if current_only:
        clauses.append("status = 'current'")
    return (" WHERE " + " AND ".join(clauses)) if clauses else "", params


def vector_search(conn, query_vec, k=5, groups=None, current_only=False):
    where, params = _where(groups, current_only)
    q = np.array(query_vec, dtype=np.float32)
    rows = conn.execute(
        f"SELECT id, doc_id, heading, content, 1 - (embedding <=> %s) AS similarity"
        f" FROM chunks{where} ORDER BY embedding <=> %s LIMIT %s",
        [q, *params, q, k]).fetchall()
    return [Hit(*r) for r in rows]


def keyword_search(conn, question, k=5, groups=None, current_only=False):
    where, params = _where(groups, current_only)
    where = (where + " AND " if where else " WHERE ") + "tsv @@ query"
    rows = conn.execute(
        f"SELECT id, doc_id, heading, content, ts_rank_cd(tsv, query) AS rank"
        f" FROM chunks, websearch_to_tsquery('english', %s) AS query{where}"
        f" ORDER BY rank DESC, id LIMIT %s",
        [question, *params, k]).fetchall()
    return [Hit(*r) for r in rows]


def or_query(question):
    """websearch_to_tsquery ANDs every word; for recall, OR the words instead."""
    words = [w.strip("?.,!:;\"'()").lower() for w in question.split()]
    return " or ".join(w for w in words if w)


def rrf(result_lists, k=60, top=5):
    """Reciprocal rank fusion: score = sum over lists of 1 / (k + rank)."""
    scores, best = {}, {}
    for hits in result_lists:
        for rank, h in enumerate(hits, start=1):
            scores[h.chunk_id] = scores.get(h.chunk_id, 0.0) + 1.0 / (k + rank)
            best.setdefault(h.chunk_id, h)
    order = sorted(scores, key=lambda cid: (-scores[cid], cid))[:top]
    return [Hit(best[c].chunk_id, best[c].doc_id, best[c].heading, best[c].content, scores[c]) for c in order]


def hybrid_search(conn, embedder, question, k=5, pool=20, groups=None, current_only=False):
    vec = vector_search(conn, embedder.embed_query(question), pool, groups, current_only)
    kw = keyword_search(conn, or_query(question), pool, groups, current_only)
    return rrf([vec, kw], top=k)
