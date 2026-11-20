"""Answer every labelled question once and save the results for the answer evaluation."""
import json

from policy_assistant.assistant import PolicyAssistant

bot = PolicyAssistant()
results = []
for q in json.load(open("questions.json")):
    r = bot.ask(q["q"], groups=[q["role"]])
    results.append({"id": q["id"], "question": q["q"], "answer": r["answer"], "check": r["check"],
                    "sources": [{"doc_id": s.doc_id, "content": s.content} for s in r["sources"]],
                    "usage": r["usage"], "timings": r["timings"]})
    print(f"{q['id']}: {r['answer']}")
json.dump(results, open("l12_answers.json", "w"), indent=1)
