"""Swap providers without touching the pipeline: the same models through two different APIs."""
import numpy as np

from policy_assistant.providers import OllamaChat, OllamaEmbedder, OpenAICompatibleChat, OpenAICompatibleEmbedder

native = OllamaEmbedder()
compat = OpenAICompatibleEmbedder("http://localhost:18211/v1", "nomic-embed-text",
                                  doc_prefix="search_document: ", query_prefix="search_query: ")
text = "Report a lost or stolen laptop within 1 hour."
a = np.array(native.embed_documents([text])[0])
b = np.array(compat.embed_documents([text])[0])
print(f"{native.name}: {len(a)} numbers   {compat.name}: {len(b)} numbers")
print(f"cosine between the two vectors: {a @ b / (np.linalg.norm(a) * np.linalg.norm(b)):.4f}")

for llm in (OllamaChat(), OpenAICompatibleChat("http://localhost:18211/v1", "qwen2.5:3b")):
    reply, usage = llm.complete([{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=5)
    print(f"{llm.name}: {reply!r}  prompt tokens {usage['prompt_tokens']}")
