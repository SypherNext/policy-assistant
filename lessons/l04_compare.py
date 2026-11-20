"""How often does the right document appear in the top 3 chunks, per chunking strategy?"""
import json

import numpy as np

from policy_assistant import chunking
from policy_assistant.documents import load_all
from policy_assistant.providers import OllamaEmbedder

emb = OllamaEmbedder()
docs = [d for d in load_all() if d.status == "current"]
questions = [q for q in json.load(open("questions.json")) if q["relevant"]]
qvecs = np.array([emb.embed_query(q["q"]) for q in questions])

strategies = {
    "fixed 40 words, no overlap": lambda d: chunking.fixed_size(d, 40, 0),
    "fixed 40 words, 10 overlap": lambda d: chunking.fixed_size(d, 40, 10),
    "fixed 15 words, no overlap": lambda d: chunking.fixed_size(d, 15, 0),
    "sentences, max 60 words":    lambda d: chunking.sentences(d, 60),
    "by heading, max 120 words":  lambda d: chunking.by_heading(d, 120),
    "whole document":             lambda d: [chunking.Chunk(d.id, 0, "", d.body)],
}
print(f"{len(questions)} questions, {len(docs)} current documents\n")
print(f"{'strategy':28s} chunks  top 1 right doc  fact in top 3  words in top 3")
for name, fn in strategies.items():
    chunks = [c for d in docs for c in fn(d)]
    C = np.array(emb.embed_documents([c.text for c in chunks]))
    top1 = fact = words = 0
    for q, qv in zip(questions, qvecs):
        top = np.argsort(-(C @ qv))[:3]
        top1 += chunks[top[0]].doc_id in q["relevant"]
        text = " ".join(chunks[i].text for i in top if chunks[i].doc_id in q["relevant"])
        fact += q["fact"].lower() in text.lower()      # the answer itself was retrieved
        words += sum(len(chunks[i].text.split()) for i in top)
    n = len(questions)
    print(f"{name:28s} {len(chunks):5d}  {top1:9d}/{n}  {fact:9d}/{n}  {words / n:10.0f}")
