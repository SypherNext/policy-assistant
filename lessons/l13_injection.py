from policy_assistant import db
from policy_assistant.documents import load_all
from policy_assistant.guards import check_answer, scan_text, strip_suspicious
from policy_assistant.indexer import index_documents
from policy_assistant.prompt import GUARD_RULE, SYSTEM, build_messages
from policy_assistant.providers import OllamaChat, OllamaEmbedder
from policy_assistant.search import vector_search

conn = db.connect()
embedder, llm = OllamaEmbedder(), OllamaChat()
docs = load_all()
by_id = {d.id: d for d in docs}
questions = ["How do I set up direct billing for my coworking desk?",
             "Is there a limit on coworking costs?"]


def naive(question):
    hits = vector_search(conn, embedder.embed_query(question), 4)
    context = "\n\n".join(h.content for h in hits)
    text, _ = llm.complete([{"role": "system", "content": "Answer the question using the context."},
                            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}])
    return text


def guarded(question):
    hits = vector_search(conn, embedder.embed_query(question), 4, ["employees"], True)
    messages, _ = build_messages(question, hits, by_id, system=SYSTEM + GUARD_RULE)
    text, _ = llm.complete(messages)
    return text


print("0. Index the documents exactly as written:", index_documents(conn, docs, embedder))

print("1. Naive pipeline, document indexed as written")
for q in questions:
    a = naive(q)
    print(f"Q: {q}\nA: {a}\n   output check: {check_answer(a)}\n")

print("2. Scan every paragraph of every document before indexing")
clean_docs = []
for d in docs:
    clean, removed = strip_suspicious(d)
    clean_docs.append(clean)
    for para in removed:
        print(f"   quarantined from {d.id}: {para[:70]}...  patterns: {len(scan_text(para))}")
print("   re-index:", index_documents(conn, clean_docs, embedder))
by_id = {d.id: d for d in clean_docs}

print("\n3. Cleaned index + source-is-data rule + output check")
for q in questions:
    a = guarded(q)
    print(f"Q: {q}\nA: {a}\n   output check: {check_answer(a)}\n")
