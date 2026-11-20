"""Retrieval metrics over a labelled question set."""


def doc_ranking(hits):
    """Chunk hits -> document ids in rank order, each document once."""
    seen, order = set(), []
    for h in hits:
        if h.doc_id not in seen:
            seen.add(h.doc_id)
            order.append(h.doc_id)
    return order


def recall_at_k(ranked, relevant, k):
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def reciprocal_rank(ranked, relevant):
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def evaluate(questions, retrieve, ks=(1, 3, 5)):
    """retrieve(question_dict) -> list of hits. Returns averages and per-question details."""
    rows = []
    for q in questions:
        ranked = doc_ranking(retrieve(q))
        row = {"id": q["id"], "rr": reciprocal_rank(ranked, q["relevant"]), "top": ranked[:3]}
        for k in ks:
            row[f"recall@{k}"] = recall_at_k(ranked, q["relevant"], k)
        rows.append(row)
    n = len(rows)
    summary = {f"recall@{k}": sum(r[f"recall@{k}"] for r in rows) / n for k in ks}
    summary["MRR"] = sum(r["rr"] for r in rows) / n
    return summary, rows
