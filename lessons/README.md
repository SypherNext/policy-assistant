# Lesson scripts

These are the scripts the lessons run on the way to the capstone. Run them from the repository root (the folder above this one) as modules, so Python finds `policy_assistant/`, `policies/` and `questions.json`:

```bash
python -m lessons.l04_chunking
python -m lessons.l06_naive "How many days of annual leave do I get?"
```

They all use the finished `policy_assistant` package. Where a later lesson changed the package (the answerability gate and the source-is-data rule in `assistant.py` arrived in the capstone, the injection scanner in lesson 13), a script from an earlier lesson can print slightly different results from the lesson text. Model answers also vary a little between machines and Ollama versions.

"Needs" says which services must be running: Ollama with both models, PostgreSQL with pgvector and an index built by `ingest.py` (or by `l05_store.py`), and the reranker (downloaded into `.models/` on first use).

| Lesson | Script | What it shows | Needs |
| --- | --- | --- | --- |
| 02 What RAG Is | `l02_no_context.py` | the model answering a policy question with no documents | Ollama |
| 03 Embeddings | `l03_embeddings.py` | embedding a query and passages, cosine similarity | Ollama |
| 03 Embeddings | `l03_map.py` | same-topic versus cross-topic similarity, a 2D map (writes `l03_map.json`) | Ollama |
| 03 Embeddings | `l03_swap.py` | the same models through Ollama's API and the OpenAI-compatible API | Ollama |
| 03 Embeddings | `l03_exercises.py` | exercise answers: another query, and the effect of the task prefix | Ollama |
| 04 Chunking | `l04_chunking.py` | four chunking strategies on one document | nothing |
| 04 Chunking | `l04_compare.py` | how often the right document is in the top 3, per strategy | Ollama |
| 05 pgvector | `l05_store.py` | reset the tables and index all 30 documents | Ollama, PostgreSQL |
| 05 pgvector | `l05_query.py` | a nearest-neighbour query in SQL | Ollama, PostgreSQL (indexed) |
| 05 pgvector | `l05_bench.py` | exact search versus HNSW and IVFFlat on 20,000 synthetic vectors | PostgreSQL |
| 06 First Pipeline | `l06_naive.py` | the naive assistant and its three failures (pass questions as arguments) | Ollama, PostgreSQL (indexed) |
| 07 Prompt Assembly | `l07_citations.py` | numbered sources and citation checks (pass a question) | Ollama, PostgreSQL (indexed) |
| 07 Prompt Assembly | `l07_repeat.py` | the same question five times at temperature 0 | Ollama, PostgreSQL (indexed) |
| 08 Hybrid Search | `l08_hybrid.py` | vector, keyword and fused rankings side by side (pass questions) | Ollama, PostgreSQL (indexed) |
| 09 Reranking | `l09_rerank.py` | hybrid candidates before and after the cross-encoder (pass questions) | Ollama, PostgreSQL (indexed), reranker |
| 10 Filters and Access | `l10_filters.py` | status and group filters in the SQL query | Ollama, PostgreSQL (indexed) |
| 10 Filters and Access | `l10_filter_pitfall.py` | why a selective filter on an HNSW index can return too few rows | PostgreSQL |
| 12 Evaluating Answers | `l12_answers.py` | answer every labelled question and save `l12_answers.json` | everything |
| 12 Evaluating Answers | `l12_judge.py` | deterministic checks, then an LLM judge, on the saved answers | Ollama, `l12_answers.json` |
| 12 Evaluating Answers | `l12_judge_v2.py` | the fixed relevance judge, tested on known off-topic answers | Ollama, `l12_answers.json` |
| 13 Prompt Injection | `l13_injection.py` | the hostile coworking paragraph, the scanner and the guard rule | Ollama, PostgreSQL (indexed) |
| 13 Prompt Injection | `l13_exercises.py` | exercise answers: the scanner and the secret-request check | nothing |
| 14 Index Freshness | `l14_reindex.py` | edit, delete and add a document, re-index incrementally, restore | Ollama, PostgreSQL (indexed) |
| 15 Cost and Caching | `l15_tokens.py` | where the prompt's tokens go | everything |
| 15 Cost and Caching | `l15_latency.py` | where the time goes in one answer, and an exact-match answer cache | everything |
| 15 Cost and Caching | `l15_semantic_cache.py` | why a semantic cache is risky for policies | Ollama |

The lesson 10 leak test (`l10_leaktest.py`), the lesson 11 retrieval evaluation (`l11_eval.py`) and the lesson 16 monitor (`l16_monitor.py`) are part of the finished app, so they live in the repository root.
