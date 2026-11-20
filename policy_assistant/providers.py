"""Embedding and chat providers behind two tiny interfaces.

Every class has the same methods, so the rest of the code never knows
which provider it is talking to.
"""
import httpx

from . import config


class OllamaEmbedder:
    """nomic-embed-text expects a task prefix on every input."""

    def __init__(self, url=config.OLLAMA_URL, model=config.EMBED_MODEL, dim=768):
        self.url, self.model, self.dim = url, model, dim
        self.name = f"ollama/{model}"

    def _embed(self, inputs):
        r = httpx.post(f"{self.url}/api/embed", json={"model": self.model, "input": inputs}, timeout=300)
        r.raise_for_status()
        return r.json()["embeddings"]

    def embed_documents(self, texts):
        return self._embed(["search_document: " + t for t in texts])

    def embed_query(self, text):
        return self._embed(["search_query: " + text])[0]


class OpenAICompatibleEmbedder:
    """Any server that speaks the OpenAI embeddings API: Ollama's /v1, or a hosted provider."""

    def __init__(self, base_url, model, api_key="unused", dim=768, doc_prefix="", query_prefix=""):
        self.base_url, self.model, self.dim = base_url.rstrip("/"), model, dim
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.doc_prefix, self.query_prefix = doc_prefix, query_prefix
        self.name = f"openai-compatible/{model}"

    def _embed(self, inputs):
        r = httpx.post(f"{self.base_url}/embeddings", headers=self.headers,
                       json={"model": self.model, "input": inputs}, timeout=300)
        r.raise_for_status()
        data = sorted(r.json()["data"], key=lambda d: d["index"])
        return [d["embedding"] for d in data]

    def embed_documents(self, texts):
        return self._embed([self.doc_prefix + t for t in texts])

    def embed_query(self, text):
        return self._embed([self.query_prefix + text])[0]


class OllamaChat:
    def __init__(self, url=config.OLLAMA_URL, model=config.CHAT_MODEL, num_ctx=4096):
        self.url, self.model, self.num_ctx = url, model, num_ctx
        self.name = f"ollama/{model}"

    def complete(self, messages, temperature=0.0, seed=42, max_tokens=300):
        r = httpx.post(f"{self.url}/api/chat", timeout=900, json={
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "seed": seed,
                        "num_ctx": self.num_ctx, "num_predict": max_tokens},
        })
        r.raise_for_status()
        body = r.json()
        usage = {
            "prompt_tokens": body.get("prompt_eval_count", 0),
            "output_tokens": body.get("eval_count", 0),
            "prompt_ms": body.get("prompt_eval_duration", 0) / 1e6,
            "output_ms": body.get("eval_duration", 0) / 1e6,
            "total_ms": body.get("total_duration", 0) / 1e6,
        }
        return body["message"]["content"].strip(), usage


class OpenAICompatibleChat:
    def __init__(self, base_url, model, api_key="unused"):
        self.base_url, self.model = base_url.rstrip("/"), model
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.name = f"openai-compatible/{model}"

    def complete(self, messages, temperature=0.0, seed=42, max_tokens=300):
        r = httpx.post(f"{self.base_url}/chat/completions", headers=self.headers, timeout=900, json={
            "model": self.model, "messages": messages, "temperature": temperature,
            "seed": seed, "max_tokens": max_tokens,
        })
        r.raise_for_status()
        body = r.json()
        u = body.get("usage", {})
        usage = {"prompt_tokens": u.get("prompt_tokens", 0), "output_tokens": u.get("completion_tokens", 0)}
        return body["choices"][0]["message"]["content"].strip(), usage
