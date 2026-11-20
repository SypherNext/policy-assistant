"""Ask the same question five times with temperature 0 and a fixed seed."""
from policy_assistant import db
from policy_assistant.documents import load_all
from policy_assistant.prompt import build_messages
from policy_assistant.providers import OllamaChat, OllamaEmbedder
from policy_assistant.search import vector_search

conn = db.connect()
docs = {d.id: d for d in load_all()}
embedder, llm = OllamaEmbedder(), OllamaChat()
question = "How many days of annual leave do I get?"
hits = vector_search(conn, embedder.embed_query(question), k=4)
messages, sources = build_messages(question, hits, docs)

for run in range(1, 6):
    text, _ = llm.complete(messages, temperature=0.0, seed=42)
    print(f"run {run}: {text}")
