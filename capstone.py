"""Capstone: the Policy Assistant from the command line."""
import argparse

from policy_assistant.assistant import PolicyAssistant
from policy_assistant.guards import check_answer

parser = argparse.ArgumentParser()
parser.add_argument("question")
parser.add_argument("--role", choices=["employees", "managers"], default="employees")
args = parser.parse_args()

bot = PolicyAssistant()
r = bot.ask(args.question, groups=[args.role])
answer = r["answer"]
if check_answer(answer)["asks_for_secret"]:
    answer = "This answer was withheld because it asked for confidential information."
print(answer)
print("\nSources:")
for i, s in enumerate(r["sources"], 1):
    d = bot.docs[s.doc_id]
    print(f"  [{i}] {d.title}, version {d.version} ({s.heading.split(' > ')[-1]})")
print(f"\ncheck: {r['check']}")
print(f"retrieve {r['timings']['retrieve_ms']:.0f} ms, answer {r['timings']['llm_ms'] / 1000:.1f} s")
