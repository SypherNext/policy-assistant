"""Exact search versus HNSW and IVFFlat on 20,000 synthetic 768-dimension vectors."""
import time

import numpy as np

from policy_assistant import db

rng = np.random.default_rng(7)
N, DIM, NQ, K = 20_000, 768, 50, 10
centres = rng.normal(size=(1000, DIM))
data = centres[rng.integers(0, 1000, N)] + rng.normal(scale=1.0, size=(N, DIM))
data /= np.linalg.norm(data, axis=1, keepdims=True)
queries = centres[rng.integers(0, 1000, NQ)] + rng.normal(scale=1.0, size=(NQ, DIM))
queries /= np.linalg.norm(queries, axis=1, keepdims=True)
truth = [set(np.argsort(-(data @ q))[:K] + 1) for q in queries]     # ids start at 1

conn = db.connect()
conn.execute("DROP TABLE IF EXISTS bench")
conn.execute("CREATE TABLE bench (id int PRIMARY KEY, embedding vector(768))")
with conn.cursor() as cur, cur.copy("COPY bench (id, embedding) FROM STDIN WITH (FORMAT BINARY)") as copy:
    copy.set_types(["int4", "vector"])
    for i, v in enumerate(data.astype(np.float32), start=1):
        copy.write_row((i, v))
conn.execute("SET maintenance_work_mem = '512MB'")


SQL = "SELECT id FROM bench ORDER BY embedding <=> %s LIMIT %s"


def run(label):
    plan = " ".join(r[0] for r in conn.execute("EXPLAIN " + SQL, (queries[0].astype(np.float32), K)))
    used = "index scan" if "Index Scan" in plan else "full scan"
    recalls, times = [], []
    for q, t in zip(queries.astype(np.float32), truth):
        start = time.perf_counter()
        rows = conn.execute(SQL, (q, K)).fetchall()
        times.append((time.perf_counter() - start) * 1000)
        recalls.append(len(t & {r[0] for r in rows}) / K)
    print(f"  {label:22s} {used:10s}  recall@10 {np.mean(recalls):.3f}  {np.median(times):6.2f} ms per query (median)")


print(f"{N:,} vectors, {DIM} dimensions, {NQ} queries")
run("exact (no index)")

start = time.perf_counter()
conn.execute("CREATE INDEX bench_hnsw ON bench USING hnsw (embedding vector_cosine_ops)")
size = conn.execute("SELECT pg_size_pretty(pg_relation_size('bench_hnsw'))").fetchone()[0]
print(f"\nHNSW build (m=16, ef_construction=64): {time.perf_counter() - start:.1f} s, {size}")
for ef in (10, 20, 40, 100, 200):
    conn.execute(f"SET hnsw.ef_search = {ef}")
    run(f"HNSW ef_search={ef}")
conn.execute("DROP INDEX bench_hnsw")

start = time.perf_counter()
conn.execute("CREATE INDEX bench_ivf ON bench USING ivfflat (embedding vector_cosine_ops) WITH (lists = 20)")
size = conn.execute("SELECT pg_size_pretty(pg_relation_size('bench_ivf'))").fetchone()[0]
print(f"\nIVFFlat build (lists=20): {time.perf_counter() - start:.1f} s, {size}")
for probes in (1, 2, 5, 10):
    conn.execute(f"SET ivfflat.probes = {probes}")
    run(f"IVFFlat probes={probes}")
conn.execute("DROP TABLE bench")
