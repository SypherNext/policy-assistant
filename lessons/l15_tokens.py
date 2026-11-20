"""Split the prompt's token count into rules, sources and question."""
from policy_assistant import db
from policy_assistant.documents import load_all
from policy_assistant.prompt import SYSTEM, build_messages
from policy_assistant.providers import OllamaChat, OllamaEmbedder
from policy_assistant.rerank import CrossEncoderReranker
from policy_assistant.search import hybrid_search

conn = db.connect()
llm, emb, rr = OllamaChat(), OllamaEmbedder(), CrossEncoderReranker()
docs = {d.id: d for d in load_all()}
q = "What is the hotel limit in Paris?"
hits = rr.rerank(q, hybrid_search(conn, emb, q, 20, 20, ["employees"], True), top=4)


def count(messages):
    return llm.complete(messages, max_tokens=1)[1]["prompt_tokens"]


full = count(build_messages(q, hits, docs)[0])
no_sources = count(build_messages(q, [], docs)[0])
only_system = count([{"role": "system", "content": SYSTEM}, {"role": "user", "content": ""}])
print(f"full prompt:            {full} tokens")
print(f"rules + question only:  {no_sources} tokens")
print(f"rules only:             {only_system} tokens")
print(f"so the 4 sources are about {full - no_sources} tokens ({(full - no_sources) / full:.0%})")
