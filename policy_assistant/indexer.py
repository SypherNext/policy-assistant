"""Incremental indexer: re-embed only documents whose text or embedding model changed."""
import numpy as np

from . import chunking, db


def index_documents(conn, docs, embedder, chunker=chunking.by_heading, cleaner=None):
    stats = {"added": 0, "changed": 0, "unchanged": 0, "deleted": 0, "chunks_embedded": 0}
    if cleaner:                                   # e.g. guards.strip_suspicious
        cleaned = [cleaner(d) for d in docs]
        docs = [d for d, _ in cleaned]
        stats["quarantined"] = sum(len(r) for _, r in cleaned)
    known = {row[0]: (row[1], row[2]) for row in
             conn.execute("SELECT id, content_hash, embed_model FROM documents")}
    for doc in docs:
        if known.get(doc.id) == (doc.content_hash, embedder.name):
            stats["unchanged"] += 1
            continue
        stats["changed" if doc.id in known else "added"] += 1
        chunks = chunker(doc)
        vectors = embedder.embed_documents([c.text for c in chunks])
        with conn.transaction():                      # readers see the old or the new version, never half
            conn.execute("DELETE FROM documents WHERE id = %s", (doc.id,))   # cascades to chunks
            conn.execute(
                "INSERT INTO documents (id, title, category, version, status, audience, content_hash, embed_model)"
                " VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                (doc.id, doc.title, doc.category, doc.version, doc.status, doc.audience,
                 doc.content_hash, embedder.name))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (doc_id, chunk_no, heading, content, status, allowed_groups, embedding)"
                    " VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    [(doc.id, c.chunk_no, c.heading, c.text, doc.status, db.groups_for(doc.audience),
                      np.array(v, dtype=np.float32)) for c, v in zip(chunks, vectors)])
        stats["chunks_embedded"] += len(chunks)
    gone = set(known) - {d.id for d in docs}
    for doc_id in gone:
        conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
    stats["deleted"] = len(gone)
    return stats
