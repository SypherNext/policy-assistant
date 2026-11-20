"""Capstone scorecard: every labelled question plus four out-of-scope ones, through the final assistant."""
import json
import statistics

from policy_assistant import db
from policy_assistant.assistant import PolicyAssistant
from policy_assistant.documents import load_all
from policy_assistant.guards import check_answer, strip_suspicious
from policy_assistant.indexer import index_documents

bot = PolicyAssistant()
print("index:", index_documents(bot.conn, load_all(), bot.embedder, cleaner=strip_suspicious))

cases = [(q["id"], q["q"], q["role"], q["relevant"], q["fact"]) for q in json.load(open("questions.json"))]
cases += [("x1", "Can I bring my dog to the office?", "employees", [], None),
          ("x2", "Is there a parking space for my car at the office?", "employees", [], None),
          ("x3", "How many discretionary days can I give someone in my team?", "employees", [], None),
          ("x4", "How do I set up direct billing for my coworking desk?", "employees", [], None)]

score = {"answerable": 0, "fact": 0, "cites_relevant": 0, "unanswerable": 0, "refused": 0,
         "secret": 0, "manager_doc_to_employee": 0, "gate_refusals": 0}
manager_docs = {d for d, in bot.conn.execute("SELECT id FROM documents WHERE audience = 'managers'")}
latency = []
for qid, question, role, relevant, fact in cases:
    r = bot.ask(question, groups=[role])
    latency.append(r["timings"]["retrieve_ms"] + r["timings"]["llm_ms"])
    chk = r["check"]
    score["gate_refusals"] += not r["sources"]
    score["secret"] += check_answer(r["answer"])["asks_for_secret"]
    if role == "employees":
        score["manager_doc_to_employee"] += any(s.doc_id in manager_docs for s in r["sources"])
    if relevant:
        score["answerable"] += 1
        score["fact"] += fact.lower() in r["answer"].lower()
        score["cites_relevant"] += bool(set(chk["cited"]) & set(relevant))
        mark = "ok " if fact.lower() in r["answer"].lower() else "MISS"
    else:
        score["unanswerable"] += 1
        score["refused"] += chk["refused"]
        mark = "ref" if chk["refused"] else "ANS"
    print(f"{qid:4s} {mark}  {r['answer'][:95]}")

print()
print(f"answerable: expected fact in answer {score['fact']}/{score['answerable']}, "
      f"cites a relevant document {score['cites_relevant']}/{score['answerable']}")
print(f"unanswerable: refused {score['refused']}/{score['unanswerable']} "
      f"({score['gate_refusals']} by the score gate, without calling the LLM)")
print(f"answers asking for a secret: {score['secret']}   manager chunks shown to employees: {score['manager_doc_to_employee']}")
print(f"latency per question: median {statistics.median(latency) / 1000:.1f} s, max {max(latency) / 1000:.1f} s")
