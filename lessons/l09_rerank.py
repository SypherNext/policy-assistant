import sys
import time

from policy_assistant import db
from policy_assistant.providers import OllamaEmbedder
from policy_assistant.rerank import CrossEncoderReranker
from policy_assistant.search import hybrid_search

conn = db.connect()
embedder = OllamaEmbedder()
t = time.perf_counter()
reranker = CrossEncoderReranker()
print(f"reranker loaded in {time.perf_counter() - t:.1f} s\n")

for question in sys.argv[1:]:
    candidates = hybrid_search(conn, embedder, question, k=20)          # wide net
    t = time.perf_counter()
    best = reranker.rerank(question, candidates, top=4)                 # careful judge
    ms = (time.perf_counter() - t) * 1000
    print(f"Q: {question}")
    print("  hybrid top 4:  " + ", ".join(h.doc_id for h in candidates[:4]))
    print(f"  reranked ({len(candidates)} candidates, {ms:.0f} ms):")
    for h in best:
        print(f"    {h.score:7.3f}  {h.doc_id}  ({h.heading.split(' > ')[-1]})")
    print()
