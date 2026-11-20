"""Cross-encoder reranking with a small local model (runs on CPU through ONNX Runtime)."""
import os

from dataclasses import replace

from fastembed.rerank.cross_encoder import TextCrossEncoder

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".models")


class CrossEncoderReranker:
    def __init__(self, model="Xenova/ms-marco-MiniLM-L-6-v2"):
        self.model = TextCrossEncoder(model_name=model, cache_dir=CACHE)
        self.name = model

    def rerank(self, question, hits, top=4):
        scores = list(self.model.rerank(question, [h.content for h in hits]))
        ranked = sorted(zip(hits, scores), key=lambda p: -p[1])[:top]
        return [replace(h, score=float(s)) for h, s in ranked]
