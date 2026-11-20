import time

from policy_assistant import db
from policy_assistant.documents import load_all
from policy_assistant.indexer import index_documents
from policy_assistant.providers import OllamaEmbedder

conn = db.connect()
db.reset(conn)                                   # drop and recreate the two tables
emb = OllamaEmbedder()

t = time.perf_counter()
stats = index_documents(conn, load_all(), emb)
print(f"indexed in {time.perf_counter() - t:.1f} s: {stats}")

print(conn.execute("SELECT count(*) FROM documents").fetchone()[0], "documents,",
      conn.execute("SELECT count(*) FROM chunks").fetchone()[0], "chunks")
for name, size in conn.execute(
        "SELECT relname, pg_size_pretty(CASE WHEN relkind = 'r' THEN pg_table_size(oid) ELSE pg_relation_size(oid) END) FROM pg_class"
        " WHERE relname IN ('chunks', 'chunks_embedding_hnsw', 'chunks_tsv_gin') ORDER BY relname"):
    print(f"  {name:24s} {size}")
