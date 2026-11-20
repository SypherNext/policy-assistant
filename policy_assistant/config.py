import os

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:18211")
PG_DSN = os.environ.get("PG_DSN", "postgresql://rag:ragpass@localhost:18232/policies")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "qwen2.5:3b")
POLICY_DIR = os.environ.get(
    "POLICY_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "policies")
)
