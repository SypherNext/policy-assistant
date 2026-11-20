"""Index the policy documents: create the schema if needed, then run the incremental indexer
with the injection scanner, as lessons 05, 13 and 14 do. Run it again after you edit, add or
delete a policy; only the changed documents are re-embedded.

python ingest.py           index what changed
python ingest.py --reset   drop both tables and index everything from scratch
"""
import argparse
import time

from policy_assistant import db
from policy_assistant.documents import load_all
from policy_assistant.guards import strip_suspicious
from policy_assistant.indexer import index_documents
from policy_assistant.providers import OllamaEmbedder

parser = argparse.ArgumentParser()
parser.add_argument("--reset", action="store_true", help="drop and recreate the two tables first")
args = parser.parse_args()

conn = db.connect()
if args.reset:
    db.reset(conn)
else:
    conn.execute(db.SCHEMA)                    # CREATE ... IF NOT EXISTS: safe to run every time

t = time.perf_counter()
stats = index_documents(conn, load_all(), OllamaEmbedder(), cleaner=strip_suspicious)
print(f"indexed in {time.perf_counter() - t:.1f} s: {stats}")
print(conn.execute("SELECT count(*) FROM documents").fetchone()[0], "documents,",
      conn.execute("SELECT count(*) FROM chunks").fetchone()[0], "chunks")
