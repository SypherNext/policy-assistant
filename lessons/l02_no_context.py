from policy_assistant.providers import OllamaChat

llm = OllamaChat()
question = "How many days of annual leave do I get at Larkfield?"
answer, usage = llm.complete([{"role": "user", "content": question}], max_tokens=120)
print(f"Q: {question}\nA: {answer}")
