"""Fix the relevance judge: reason first, verdict last, and test it on answers known to be off-topic."""
import json
import re

from policy_assistant.providers import OllamaChat

judge = OllamaChat()
answers = json.load(open("l12_answers.json"))

JUDGE_RELEVANT_V2 = """You grade whether an answer responds to a question. Ignore whether the answer is true.
A short answer that addresses the question counts as relevant. An answer about a different topic is not relevant.

Question: {question}
Answer: {answer}

Write one sentence of reasoning. Then write a last line that is exactly VERDICT: YES or VERDICT: NO."""


def relevant(question, answer):
    text, _ = judge.complete([{"role": "user", "content": JUDGE_RELEVANT_V2.format(question=question, answer=answer)}],
                             max_tokens=80)
    m = re.search(r"VERDICT:\s*(YES|NO)", text, re.IGNORECASE)
    return m.group(1).upper() if m else "UNPARSED"


on_topic = [relevant(a["question"], a["answer"]) for a in answers]
print(f"real answers judged relevant: {on_topic.count('YES')}/{len(answers)}"
      f"  (NO: {on_topic.count('NO')}, unparsed: {on_topic.count('UNPARSED')})")

# Pair each of 10 questions with the answer to a different question: all should be NO.
off = [relevant(answers[i]["question"], answers[(i + 7) % len(answers)]["answer"]) for i in range(10)]
print(f"swapped (off-topic) answers judged relevant: {off.count('YES')}/10  (NO: {off.count('NO')})")
print("example swap:", answers[0]["question"], "->", answers[7]["answer"][:70], "->", off[0])
