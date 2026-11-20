import json

import numpy as np

from policy_assistant.providers import OllamaEmbedder

emb = OllamaEmbedder()
passages = {
    "leave": ["Full-time employees get 25 days of paid annual leave per year.",
              "You can carry up to 5 unused leave days into the next year.",
              "Tell your manager before 10:00 if you are too ill to work."],
    "expenses": ["The hotel limit is 180 EUR per night, 250 EUR in high-cost cities.",
                 "Claim 0.30 EUR per kilometre when you use your own car.",
                 "Submit every expense claim within 30 days of the purchase."],
    "security": ["Report a lost or stolen laptop within 1 hour.",
                 "Passwords must be at least 14 characters long.",
                 "Report suspicious emails with the Report phishing button."],
    "remote": ["Hybrid employees work from home up to 3 days a week.",
               "You can work from another country for up to 20 days a year.",
               "Fully remote employees get 30 EUR per month for internet."],
}
queries = ["Can I take my unused holiday into January?",
           "My laptop was stolen at the airport, what now?"]

labels = [(topic, s) for topic, group in passages.items() for s in group]
D = np.array(emb.embed_documents([s for _, s in labels]))
Q = np.array([emb.embed_query(q) for q in queries])

for q, qv in zip(queries, Q):
    sims = D @ qv                  # every vector has length 1, so the dot product is the cosine
    order = np.argsort(-sims)
    print(f"query: {q}")
    for j in order[:3]:
        print(f"  {sims[j]:.3f}  [{labels[j][0]}] {labels[j][1]}")
    print(f"  lowest: {sims[order[-1]]:.3f}  [{labels[order[-1]][0]}] {labels[order[-1]][1]}")
    print()

S = D @ D.T
within = [S[i, j] for i in range(12) for j in range(i + 1, 12) if labels[i][0] == labels[j][0]]
across = [S[i, j] for i in range(12) for j in range(i + 1, 12) if labels[i][0] != labels[j][0]]
print(f"average cosine between passages on the same topic:   {np.mean(within):.3f}")
print(f"average cosine between passages on different topics: {np.mean(across):.3f}")

# Squash 768 dimensions into 2 for a picture (principal component analysis).
allv = np.vstack([D, Q])
X = allv - allv.mean(axis=0)
_, _, vt = np.linalg.svd(X, full_matrices=False)
xy = X @ vt[:2].T
points = [{"kind": "doc", "topic": t, "text": s} for t, s in labels] + \
         [{"kind": "query", "topic": "query", "text": q} for q in queries]
for p, (a, b) in zip(points, xy):
    p["x"], p["y"] = float(a), float(b)
json.dump(points, open("l03_map.json", "w"), indent=1)
