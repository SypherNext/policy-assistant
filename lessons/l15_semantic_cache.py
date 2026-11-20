"""Why a semantic cache (reuse the answer of a similar question) is risky for policies."""
import numpy as np

from policy_assistant.providers import OllamaEmbedder

emb = OllamaEmbedder()
pairs = [("What is the hotel limit in Paris?", "What is the hotel limit in Berlin?"),
         ("What is the hotel limit in Paris?", "what's the max hotel price per night in Paris"),
         ("Can I fly business class to New York?", "Can I fly business class to Brussels?")]
for a, b in pairs:
    va, vb = np.array(emb.embed_query(a)), np.array(emb.embed_query(b))
    print(f"{float(va @ vb):.3f}  {a!r} vs {b!r}")
