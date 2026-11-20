import sys

from policy_assistant import db
from policy_assistant.providers import OllamaEmbedder
from policy_assistant.search import keyword_search, or_query, rrf, vector_search

conn = db.connect()
embedder = OllamaEmbedder()


def show(label, hits):
    print(f"  {label}")
    for rank, h in enumerate(hits, 1):
        print(f"    {rank}. {h.score:.4f}  {h.doc_id}  ({h.heading.split(' > ')[-1]})")


for question in sys.argv[1:]:
    print(f"Q: {question}")
    print(f"  keyword query: {or_query(question)}")
    vec = vector_search(conn, embedder.embed_query(question), k=20)
    kw = keyword_search(conn, or_query(question), k=20)
    show("vector (cosine similarity)", vec[:5])
    show("keyword (ts_rank_cd)", kw[:5])
    show("hybrid (reciprocal rank fusion, k=60)", rrf([vec, kw], k=60, top=5))
    print()
