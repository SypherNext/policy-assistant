import sys

from policy_assistant import db
from policy_assistant.documents import load_all
from policy_assistant.prompt import build_messages, check_citations
from policy_assistant.providers import OllamaChat, OllamaEmbedder
from policy_assistant.search import vector_search

conn = db.connect()
docs = {d.id: d for d in load_all()}
embedder, llm = OllamaEmbedder(), OllamaChat()

for question in sys.argv[1:]:
    hits = vector_search(conn, embedder.embed_query(question), k=4)
    messages, sources = build_messages(question, hits, docs)
    text, usage = llm.complete(messages)
    print(f"Q: {question}")
    for i, s in enumerate(sources, 1):
        print(f"  [{i}] {s.doc_id} (version {docs[s.doc_id].version})")
    print(f"A: {text}")
    print(f"   check: {check_citations(text, sources)}  prompt tokens: {usage['prompt_tokens']}\n")
