"""Incremental re-indexing: change one document, delete one, add one, then restore."""
import os
import shutil
import time

from policy_assistant import db
from policy_assistant.config import POLICY_DIR
from policy_assistant.documents import load_all
from policy_assistant.guards import strip_suspicious
from policy_assistant.indexer import index_documents
from policy_assistant.providers import OllamaEmbedder
from policy_assistant.search import vector_search

conn = db.connect()
emb = OllamaEmbedder()
work = os.path.join(os.path.dirname(os.path.abspath(__file__)), "policies-work")
shutil.rmtree(work, ignore_errors=True)
shutil.copytree(POLICY_DIR, work)


def run(label, folder):
    t = time.perf_counter()
    stats = index_documents(conn, load_all(folder), emb, cleaner=strip_suspicious)
    print(f"{label:38s} {stats}  ({time.perf_counter() - t:.1f} s)")


run("1. nothing changed:", work)

path = os.path.join(work, "expenses-meals-travel.md")
text = open(path, encoding="utf-8").read()
text = text.replace("up to 60 EUR per day", "up to 65 EUR per day").replace("version: 2026.1", "version: 2026.2")
open(path, "w", encoding="utf-8").write(text)
os.remove(os.path.join(work, "leave-public-holidays.md"))
with open(os.path.join(work, "security-usb.md"), "w", encoding="utf-8") as f:
    f.write("---\nid: security-usb\ntitle: USB Drives\ncategory: security\nversion: 2026.1\n"
            "effective: 2026-11-01\nstatus: current\naudience: all\n---\n# USB Drives\n\n"
            "Do not use USB drives on company laptops. Share files through the company file storage instead.\n")
run("2. one edited, one deleted, one new:", work)

hit = vector_search(conn, emb.embed_query("meal allowance on an overnight trip"), 1)[0]
print(f"   top chunk now: {hit.doc_id}: {hit.content.splitlines()[1][:80]}")
print("   documents in the index:", conn.execute("SELECT count(*) FROM documents").fetchone()[0])

shutil.rmtree(work)
run("3. back to the original folder:", POLICY_DIR)
