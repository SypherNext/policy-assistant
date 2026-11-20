"""The naive Policy Assistant: embed, retrieve 4 chunks, paste them into a prompt."""
import sys
import time

from policy_assistant import db
from policy_assistant.providers import OllamaChat, OllamaEmbedder
from policy_assistant.search import vector_search

conn = db.connect()
embedder, llm = OllamaEmbedder(), OllamaChat()


def answer(question, k=4):
    t0 = time.perf_counter()
    hits = vector_search(conn, embedder.embed_query(question), k)
    t1 = time.perf_counter()
    context = "\n\n".join(h.content for h in hits)
    messages = [
        {"role": "system", "content": "Answer the question using the context."},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    text, usage = llm.complete(messages)
    t2 = time.perf_counter()
    return hits, text, usage, (t1 - t0) * 1000, (t2 - t1) * 1000


for question in sys.argv[1:]:
    hits, text, usage, ms_retrieve, ms_llm = answer(question)
    print(f"Q: {question}")
    for h in hits:
        print(f"   {h.score:.3f}  {h.doc_id}  ({h.heading})")
    print(f"A: {text}")
    print(f"   retrieval {ms_retrieve:.0f} ms, LLM {ms_llm / 1000:.1f} s, "
          f"{usage['prompt_tokens']} prompt tokens, {usage['output_tokens']} output tokens\n")
