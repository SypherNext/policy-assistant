"""A selective filter on top of an HNSW index: 20,000 chunks shared by 100 teams."""
import numpy as np

from policy_assistant import db

rng = np.random.default_rng(11)
N, DIM = 20_000, 768
data = rng.normal(size=(N, DIM)).astype(np.float32)
data /= np.linalg.norm(data, axis=1, keepdims=True)
team = rng.integers(0, 100, N)                 # each chunk belongs to one of 100 teams

conn = db.connect()
conn.execute("DROP TABLE IF EXISTS team_chunks")
conn.execute("CREATE TABLE team_chunks (id int PRIMARY KEY, team int, embedding vector(768))")
with conn.cursor() as cur, cur.copy("COPY team_chunks FROM STDIN WITH (FORMAT BINARY)") as copy:
    copy.set_types(["int4", "int4", "vector"])
    for i, (t, v) in enumerate(zip(team, data), start=1):
        copy.write_row((i, int(t), v))
conn.execute("SET maintenance_work_mem = '512MB'")
conn.execute("CREATE INDEX ON team_chunks USING hnsw (embedding vector_cosine_ops)")
conn.execute("ANALYZE team_chunks")

q = rng.normal(size=DIM).astype(np.float32)
SQL = "SELECT id FROM team_chunks WHERE team = 7 ORDER BY embedding <=> %s LIMIT 10"
print(f"rows for team 7: {int((team == 7).sum())} of {N:,}")

conn.execute("SET enable_seqscan = off")       # make sure the HNSW index is used, as on a big table
for setting in ("off", "relaxed_order"):
    conn.execute(f"SET hnsw.iterative_scan = {setting}")
    rows = conn.execute(SQL, (q,)).fetchall()
    print(f"hnsw.iterative_scan = {setting:13s} -> {len(rows)} rows returned (asked for 10)")
conn.execute("DROP TABLE team_chunks")
