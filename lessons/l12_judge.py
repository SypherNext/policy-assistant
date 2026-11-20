"""Score saved answers: cheap deterministic checks first, then an LLM judge."""
import json
import re

from policy_assistant.prompt import check_citations
from policy_assistant.providers import OllamaChat

judge = OllamaChat()                     # the same small model judges its own answers (a known bias)
questions = {q["id"]: q for q in json.load(open("questions.json"))}
answers = json.load(open("l12_answers.json"))

JUDGE_FAITHFUL = """You check answers from a policy assistant.
Sources:
{sources}

Answer to check:
{answer}

Is every fact in the answer supported by the sources? Reply with SUPPORTED or UNSUPPORTED on the first line, then one short sentence of reason."""

JUDGE_RELEVANT = """Question: {question}
Answer: {answer}

Does the answer respond to the question that was asked? Reply with YES or NO on the first line, then one short sentence of reason."""


def ask_judge(prompt):
    text, _ = judge.complete([{"role": "user", "content": prompt}], max_tokens=60)
    first = text.strip().split()[0].strip(".:*").upper() if text.strip() else ""
    return first, text.strip().replace("\n", " ")


def faithful(answer, sources):
    src = "\n\n".join(f"[{i}] {s['content']}" for i, s in enumerate(sources, 1))
    return ask_judge(JUDGE_FAITHFUL.format(sources=src, answer=answer))


class Src:                                # check_citations only needs .doc_id
    def __init__(self, d):
        self.doc_id = d["doc_id"]


rows = []
for a in answers:
    q = questions[a["id"]]
    chk = check_citations(a["answer"], [Src(s) for s in a["sources"]])
    if q["fact"] is None:
        fact_ok = chk["refused"]
    else:
        fact_ok = q["fact"].lower() in a["answer"].lower()
    cites_relevant = (not q["relevant"]) or bool(set(chk["cited"]) & set(q["relevant"]))
    f_verdict, f_text = faithful(a["answer"], a["sources"])
    r_verdict, r_text = ask_judge(JUDGE_RELEVANT.format(question=a["question"], answer=a["answer"]))
    rows.append({"id": a["id"], "fact_ok": fact_ok, "cites_relevant": cites_relevant,
                 "faithful": f_verdict, "relevant": r_verdict, "faithful_reason": f_text})
    print(f"{a['id']}  fact {'ok ' if fact_ok else 'MISS'}  cites relevant doc {'yes' if cites_relevant else 'NO '}"
          f"  judge faithful {f_verdict:11s}  judge relevant {r_verdict}")

n = len(rows)
print(f"\nfact found in answer:        {sum(r['fact_ok'] for r in rows)}/{n}")
print(f"cites a relevant document:   {sum(r['cites_relevant'] for r in rows)}/{n}")
print(f"judge says SUPPORTED:        {sum(r['faithful'] == 'SUPPORTED' for r in rows)}/{n}")
print(f"judge says relevant (YES):   {sum(r['relevant'] == 'YES' for r in rows)}/{n}")

# Test the judge: change the number in eight correct answers and see if it notices.
print("\nJudge test: answers with one number changed")
caught = 0
tests = [("q01", "25", "30"), ("q05", "16", "12"), ("q09", "60", "75"), ("q10", "250", "300"),
         ("q14", "80", "120"), ("q17", "14", "10"), ("q23", "10:00", "09:00"), ("q26", "150", "250")]
by_id = {a["id"]: a for a in answers}
for qid, old, new in tests:
    wrong = by_id[qid]["answer"].replace(old, new, 1)
    verdict, text = faithful(wrong, by_id[qid]["sources"])
    caught += verdict == "UNSUPPORTED"
    print(f"  {qid}  {old} -> {new}  judge: {verdict}")
print(f"caught {caught} of {len(tests)} wrong answers")
json.dump(rows, open("l12_judge.json", "w"), indent=1)
