import json

import numpy as np

from policy_assistant.providers import OllamaEmbedder

emb = OllamaEmbedder()

query = "How many holiday days do I get?"
passages = [
    "Every full-time employee gets 25 days of paid annual leave per calendar year.",
    "You can claim up to 60 EUR per day for meals on a business trip.",
    "Report a lost or stolen laptop to the security team within 1 hour.",
]

q = np.array(emb.embed_query(query))
P = np.array(emb.embed_documents(passages))

print("dimensions:", q.shape[0])
print("first 5 numbers of the query vector:", np.round(q[:5], 4))
print("length (L2 norm) of the query vector:", round(float(np.linalg.norm(q)), 4))


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


print()
print(f"query: {query}")
for text, p in zip(passages, P):
    print(f"  cosine {cosine(q, p):.3f}  {text}")
