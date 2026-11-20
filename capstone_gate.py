"""Choose an answerability threshold from data: the top reranker score per question."""
import json

from policy_assistant.assistant import PolicyAssistant

bot = PolicyAssistant()
labelled = [(q["q"], q["role"], bool(q["relevant"])) for q in json.load(open("questions.json"))]
extra = [("Can I bring my dog to the office?", "employees", False),
         ("Is there a parking space for my car at the office?", "employees", False),
         ("How many discretionary days can I give someone in my team?", "employees", False),
         ("Up to what amount can I approve an expense exception myself?", "employees", False)]

rows = []
for question, role, answerable in labelled + extra:
    top = bot.retrieve(question, [role])[0].score
    rows.append((top, answerable, role, question))

rows.sort()
print("lowest top scores (score, answerable, asked as, question):")
for top, answerable, role, question in rows[:10]:
    print(f"  {top:8.3f}  {str(answerable):5s}  {role:9s}  {question}")
answerable = [t for t, a, _, _ in rows if a]
unanswerable = [t for t, a, _, _ in rows if not a]
print(f"\nanswerable questions: {len(answerable)}, lowest top score {min(answerable):.3f}")
print(f"unanswerable questions: {len(unanswerable)}, highest top score {max(unanswerable):.3f}")
for threshold in (-4.0, -2.0, -1.0, 0.0):
    wrong_refusals = sum(t < threshold for t in answerable)
    caught = sum(t < threshold for t in unanswerable)
    print(f"threshold {threshold:5.1f}: refuses {caught}/{len(unanswerable)} unanswerable, "
          f"{wrong_refusals}/{len(answerable)} answerable")
