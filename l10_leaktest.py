"""Leak test: no employee question may retrieve a managers-only chunk."""
import json

from policy_assistant import db
from policy_assistant.providers import OllamaEmbedder
from policy_assistant.search import hybrid_search

conn = db.connect()
embedder = OllamaEmbedder()
manager_docs = {r[0] for r in conn.execute("SELECT id FROM documents WHERE audience = 'managers'")}
questions = [q["q"] for q in json.load(open("questions.json"))] + [
    "Can my manager give me extra days off after a release weekend?",
    "Who approves an expense that goes over the limit?",
    "How often are access rights reviewed?",
]
leaks = {"employees": 0, "none": 0}
for q in questions:
    for label, groups in (("employees", ["employees"]), ("none", None)):
        hits = hybrid_search(conn, embedder, q, k=20, groups=groups, current_only=True)
        leaks[label] += any(h.doc_id in manager_docs for h in hits)
print(f"manager-only documents: {sorted(manager_docs)}")
print(f"{len(questions)} questions, top 20 chunks each")
print(f"questions that retrieved a manager-only chunk, employee filter on: {leaks['employees']}")
print(f"questions that retrieved a manager-only chunk, no group filter:    {leaks['none']}")
assert leaks["employees"] == 0, "access filter leaked"
