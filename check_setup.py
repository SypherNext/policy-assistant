import httpx
import psycopg

OLLAMA = "http://localhost:18211"
DSN = "postgresql://rag:ragpass@localhost:18232/policies"

print("Ollama", httpx.get(f"{OLLAMA}/api/version").json()["version"])
for m in httpx.get(f"{OLLAMA}/api/tags").json()["models"]:
    print(f"  model {m['name']:24s} {m['size'] / 1e9:.2f} GB")

with psycopg.connect(DSN) as conn:
    print(conn.execute("SELECT version()").fetchone()[0].split(" on ")[0])
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    print("pgvector", conn.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()[0])
    print("distance:", conn.execute("SELECT '[1,0]'::vector <=> '[0.6,0.8]'::vector").fetchone()[0])
