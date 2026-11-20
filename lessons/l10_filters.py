from policy_assistant import db
from policy_assistant.providers import OllamaEmbedder
from policy_assistant.search import vector_search

conn = db.connect()
embedder = OllamaEmbedder()

EMPLOYEE = ["employees"]
MANAGER = ["managers"]


def show(label, hits):
    print(f"  {label}")
    for h in hits:
        print(f"    {h.score:.3f}  {h.doc_id}  ({h.heading.split(' > ')[-1]})")


q = "How many days of annual leave do I get?"
v = embedder.embed_query(q)
print(f"Q: {q}")
show("no filter", vector_search(conn, v, k=3))
show("current documents only", vector_search(conn, v, k=3, current_only=True))

q = "How many discretionary days can I give someone in my team?"
v = embedder.embed_query(q)
print(f"\nQ: {q}")
show("asked by an employee", vector_search(conn, v, k=3, groups=EMPLOYEE, current_only=True))
show("asked by a manager", vector_search(conn, v, k=3, groups=MANAGER, current_only=True))
