import json

from policy_assistant import db
from policy_assistant.evaluation import evaluate
from policy_assistant.providers import OllamaEmbedder
from policy_assistant.rerank import CrossEncoderReranker
from policy_assistant.search import hybrid_search, keyword_search, or_query, vector_search

conn = db.connect()
embedder, reranker = OllamaEmbedder(), CrossEncoderReranker()
questions = [q for q in json.load(open("questions.json")) if q["relevant"]]
qvec = {q["id"]: embedder.embed_query(q["q"]) for q in questions}       # embed each question once


def groups(q):
    return [q["role"]]


setups = {
    "vector, no filters": lambda q: vector_search(conn, qvec[q["id"]], 20),
    "vector": lambda q: vector_search(conn, qvec[q["id"]], 20, groups(q), True),
    "keyword": lambda q: keyword_search(conn, or_query(q["q"]), 20, groups(q), True),
    "hybrid (RRF)": lambda q: hybrid_search(conn, embedder, q["q"], 20, 20, groups(q), True),
    "hybrid + rerank": lambda q: reranker.rerank(
        q["q"], hybrid_search(conn, embedder, q["q"], 20, 20, groups(q), True), top=20),
}

print(f"{len(questions)} labelled questions\n")
print(f"{'setup':20s} recall@1  recall@3  recall@5    MRR")
details = {}
for name, retrieve in setups.items():
    summary, rows = evaluate(questions, retrieve)
    details[name] = rows
    print(f"{name:20s} {summary['recall@1']:8.3f}  {summary['recall@3']:8.3f}  {summary['recall@5']:8.3f}  {summary['MRR']:.3f}")

print("\nQuestions where the first relevant document is not ranked first:")
for name, rows in details.items():
    misses = [f"{r['id']} (RR {r['rr']:.2f})" for r in rows if r["rr"] < 1]
    print(f"  {name:20s} {', '.join(misses) if misses else 'none'}")
json.dump(details, open("l11_eval.json", "w"), indent=1)
