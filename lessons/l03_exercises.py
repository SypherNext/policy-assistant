import numpy as np

from policy_assistant.providers import OllamaEmbedder

emb = OllamaEmbedder()
passages = [
    "Every full-time employee gets 25 days of paid annual leave per calendar year.",
    "You can claim up to 60 EUR per day for meals on a business trip.",
    "Report a lost or stolen laptop to the security team within 1 hour.",
]
P = np.array(emb.embed_documents(passages))
q = np.array(emb.embed_query("Who do I call if I clicked a phishing link?"))
for text, p in zip(passages, P):
    print(f"{float(q @ p):.3f}  {text}")
print("without the query prefix:")
q2 = np.array(emb._embed(["How many holiday days do I get?"])[0])
q1 = np.array(emb.embed_query("How many holiday days do I get?"))
for text, p in zip(passages, P):
    print(f"  with {float(q1 @ p):.3f}  without {float(q2 @ p):.3f}  {text}")
