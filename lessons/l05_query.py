import re

import numpy as np

from policy_assistant import db
from policy_assistant.providers import OllamaEmbedder

conn = db.connect()
q = np.array(OllamaEmbedder().embed_query("Can I take unused holiday into next year?"), dtype=np.float32)

sql = """
SELECT doc_id, chunk_no, round((1 - (embedding <=> %s))::numeric, 3) AS similarity
FROM chunks
ORDER BY embedding <=> %s
LIMIT 4
"""
conn.execute("ANALYZE chunks")             # refresh the planner statistics
for row in conn.execute(sql, (q, q)):
    print(row)

print("\nPlan with 63 rows (the planner prefers to read the whole table):")
for (line,) in conn.execute("EXPLAIN " + sql, (q, q)):
    print(" ", re.sub(r"'\[[^\]]*\]'", "'[768 numbers]'", line))

conn.execute("SET enable_seqscan = off")         # for the demo only: force the index
print("\nPlan with sequential scans switched off:")
for (line,) in conn.execute("EXPLAIN " + sql, (q, q)):
    print(" ", re.sub(r"'\[[^\]]*\]'", "'[768 numbers]'", line))
