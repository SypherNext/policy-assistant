"""Log every request as one JSON line, then compute the numbers a dashboard shows."""
import json
import statistics
import time
import uuid

from policy_assistant.assistant import PolicyAssistant
from policy_assistant.guards import check_answer

bot = PolicyAssistant()
LOG = "l16_requests.jsonl"
traffic = [
    ("employees", "How many days of annual leave do I get?"),
    ("employees", "What is the hotel limit in Paris?"),
    ("employees", "Can I bring my dog to the office?"),
    ("employees", "What is the company's share price?"),
    ("employees", "How do I set up direct billing for my coworking desk?"),
    ("managers", "Up to what amount can I approve an expense exception myself?"),
    ("employees", "Do I need a doctor's note if I am off sick for 2 days?"),
    ("employees", "Can I use SMS codes for two-factor login?"),
    ("employees", "Is there a parking space for my car at the office?"),
    ("employees", "What are the core hours?"),
]


def logged_ask(group, question):
    t = time.perf_counter()
    r = bot.ask(question, groups=[group])
    record = {
        "request_id": uuid.uuid4().hex[:8], "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "groups": [group], "question": question,
        "retrieved": [{"doc_id": s.doc_id, "score": round(s.score, 3)} for s in r["sources"]],
        "top_score": round(r["sources"][0].score, 3) if r["sources"] else None,
        "refused": r["check"]["refused"], "uncited": r["check"]["uncited"],
        "invalid_citations": r["check"]["invalid"],
        "asks_for_secret": check_answer(r["answer"])["asks_for_secret"],
        "prompt_tokens": r["usage"]["prompt_tokens"], "output_tokens": r["usage"]["output_tokens"],
        "retrieve_ms": round(r["timings"]["retrieve_ms"]), "llm_ms": round(r["timings"]["llm_ms"]),
        "total_ms": round((time.perf_counter() - t) * 1000),
        "models": {"embed": bot.embedder.name, "rerank": bot.reranker.name, "llm": bot.llm.name},
    }
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record, r["answer"]


open(LOG, "w").close()
for group, q in traffic:
    rec, answer = logged_ask(group, q)
    print(f"{rec['total_ms']:6d} ms  top score {rec['top_score']:7.3f}  {q}\n          -> {answer}")

rows = [json.loads(line) for line in open(LOG, encoding="utf-8")]
lat = sorted(r["total_ms"] for r in rows)
p95 = lat[min(len(lat) - 1, round(0.95 * (len(lat) - 1)))]
LOW = 0.0                                   # top reranker score below this: retrieval probably missed
print(f"\nrequests: {len(rows)}")
print(f"latency p50 {statistics.median(lat) / 1000:.1f} s, p95 {p95 / 1000:.1f} s")
print(f"refused: {sum(r['refused'] for r in rows)}   uncited: {sum(r['uncited'] for r in rows)}   "
      f"invalid citations: {sum(bool(r['invalid_citations']) for r in rows)}   "
      f"asks for secret: {sum(r['asks_for_secret'] for r in rows)}")
low = [r for r in rows if r["top_score"] is not None and r["top_score"] < LOW]
print(f"top reranker score below {LOW}: {len(low)}")
for r in low:
    print(f"  {r['top_score']:7.3f}  refused={r['refused']}  {r['question']}")
print(f"tokens per request: prompt median {statistics.median(r['prompt_tokens'] for r in rows):.0f}, "
      f"output median {statistics.median(r['output_tokens'] for r in rows):.0f}")
print("\none log line:")
print(json.dumps(rows[1], indent=1))
