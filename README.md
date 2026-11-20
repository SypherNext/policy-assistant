# Policy Assistant

The Policy Assistant answers employees' questions from 30 fictional policy documents of a company called Larkfield, with cited, access-controlled answers. It stores chunks and vectors in PostgreSQL with pgvector, retrieves with hybrid search (vector plus full-text, fused with reciprocal rank fusion), reranks with a small cross-encoder, refuses clearly unanswerable questions before calling the model, keeps manager-only documents away from employees, quarantines instructions hidden in documents, checks every citation, and logs each request for a dashboard. Everything runs locally on a CPU with open models served by Ollama. It is the companion code for the Sypher course "RAG in Production".

## Set up your computer

You need Git, Python 3.12 or newer, Docker, and Ollama with two models. Plan for about 8 GB of free disk (images and models) and 8 GB of RAM. No graphics card is needed; everything runs on the CPU, just slower.

### Git

| System | Install |
| --- | --- |
| Windows | `winget install --id Git.Git -e` |
| macOS | `brew install git` (or run `xcode-select --install`) |
| Ubuntu or Debian | `sudo apt update && sudo apt install -y git` |
| Fedora | `sudo dnf install -y git` |

Installers for every system are also on the Git downloads page, https://git-scm.com/downloads. Check it in a new terminal:

```bash
git --version
```

It prints a line such as `git version 2.51.0`.

### Python 3.12 or newer, with venv

The course was run on Python 3.14.3.

| System | Install |
| --- | --- |
| Windows | `winget install Python.Python.3.14` |
| macOS | `brew install python@3.14` |
| Ubuntu or Debian | `sudo apt install -y python3 python3-venv` (Ubuntu 24.04 ships Python 3.12) |
| Fedora | `sudo dnf install -y python3` |

The Python downloads page is https://www.python.org/downloads/. On Windows, tick "Add python.exe to PATH" if you use the installer from there. Check it (on Windows the command is `python`, on macOS and Linux `python3`):

```bash
python3 --version
python3 -m venv --help
```

The first command prints a line such as `Python 3.14.3`; the version must be 3.12 or higher. The second prints the venv usage text. If it says `No module named venv`, install the `python3-venv` package on Debian or Ubuntu.

### Docker

Docker runs PostgreSQL with pgvector, and in the course also Ollama.

| System | Install |
| --- | --- |
| Windows | `winget install Docker.DockerDesktop`, then start Docker Desktop |
| macOS | `brew install --cask docker`, then start Docker Desktop from Applications |
| Linux | Docker Engine: `curl -fsSL https://get.docker.com \| sh`, then `sudo usermod -aG docker $USER` and log out and back in |

The Docker Desktop page, https://www.docker.com/products/docker-desktop/, and the Docker Engine install guide, https://docs.docker.com/engine/install/, cover the same steps by hand. Check that the engine is running, not just installed:

```bash
docker version
```

The output has a `Client:` section and a `Server:` section. If the `Server:` section is missing and you see "Cannot connect to the Docker daemon", start Docker Desktop (or `sudo systemctl start docker` on Linux) and try again.

### Ollama and the two models

Ollama is a server that downloads open models and runs them behind a small HTTP API. The course runs it in a Docker container on host port 18211, which is the default in `policy_assistant/config.py`. The ports in this project (18211 and 18232) are deliberately unusual so they do not clash with an Ollama or PostgreSQL you may already run.

**Option A, as in the course: Ollama in Docker.** The named volume keeps the downloaded models when the container restarts:

```bash
docker network create rag-net
docker volume create rag-ollama-models
docker run -d --name rag-ollama --network rag-net -p 18211:11434 -v rag-ollama-models:/root/.ollama ollama/ollama
docker exec rag-ollama ollama pull nomic-embed-text
docker exec rag-ollama ollama pull qwen2.5:3b
docker exec rag-ollama ollama list
```

**Option B: Ollama installed on your computer.**

| System | Install |
| --- | --- |
| Windows | `winget install Ollama.Ollama` |
| macOS | `brew install ollama`, then `ollama serve` in its own terminal (or install the app from the download page) |
| Linux | `curl -fsSL https://ollama.com/install.sh \| sh` |

The Ollama download page is https://ollama.com/download. Then pull the models:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:3b
ollama list
```

A native Ollama listens on its standard port 11434, so tell the code where it is in every terminal you use (macOS and Linux, then Windows PowerShell):

```bash
export OLLAMA_URL=http://localhost:11434
```

```powershell
$env:OLLAMA_URL = "http://localhost:11434"
```

Either way, `ollama list` must show two lines, `qwen2.5:3b` at about 1.9 GB and `nomic-embed-text:latest` at about 274 MB, each with an ID and the time you pulled it.

- **nomic-embed-text** (274 MB) is the embedding model: it turns a piece of text into 768 numbers that capture its meaning.
- **qwen2.5:3b** (1.9 GB) is the language model that writes the answers. It is released under the Qwen Research License, not an open-source licence like most other Qwen 2.5 sizes, so read the licence before you use it in a product. For learning, it is fine. The code is provider-neutral; see "Swapping In Hosted Models" in the capstone lesson to use a different model.

### Get the code

```bash
git clone <policy-assistant repository URL> policy-assistant
cd policy-assistant
```

If you do not use Git, download the repository as a ZIP file from its page, unpack it and open a terminal in the `policy-assistant` folder.

## Run the app

Run every command from the `policy-assistant` folder. Ollama must be running with both models (see above).

1. **Create the virtual environment and install the packages.** On Windows use `python`, on macOS and Linux `python3`, for the first command:

   ```bash
   python -m venv .venv
   ```

   Activate it. In Windows PowerShell: `.venv\Scripts\Activate.ps1` (if PowerShell refuses to run scripts, first run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`). In Git Bash: `source .venv/Scripts/activate`. On macOS and Linux: `source .venv/bin/activate`. Your prompt now starts with `(.venv)`, and `python` means the environment's Python in every system. Then:

   ```bash
   pip install -r requirements.txt
   ```

   This installs psycopg, pgvector, numpy, httpx and fastembed at the versions the course used (psycopg 3.3.7, pgvector 0.5.1, numpy 2.5.4, httpx 0.28.1, fastembed 0.9.1).

2. **Configure, if you changed anything.** The defaults in `policy_assistant/config.py` match the commands in this README. `.env.example` lists the five settings (`OLLAMA_URL`, `PG_DSN`, `EMBED_MODEL`, `CHAT_MODEL`, `POLICY_DIR`). The code reads them from environment variables and does not load a `.env` file, so set any you change in your shell with `export NAME=value` (PowerShell: `$env:NAME = "value"`).

3. **Start PostgreSQL 18 with pgvector** on host port 18232. If you skipped Option A above, create the network first with `docker network create rag-net`.

   ```bash
   docker volume create rag-pgdata
   docker run -d --name rag-pg --network rag-net -p 18232:5432 -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=ragpass -e POSTGRES_DB=policies -v rag-pgdata:/var/lib/postgresql pgvector/pgvector:pg18
   ```

   The password is a throwaway for a local container. In anything shared, it comes from a secret store or an environment variable, never from code.

4. **Check the whole stack with one script:**

   ```bash
   python check_setup.py
   ```

   Expected output (your versions may be newer):

   ```text
   Ollama 0.40.14
     model qwen2.5:3b               1.93 GB
     model nomic-embed-text:latest  0.27 GB
   PostgreSQL 18.6 (Debian 18.6-1.pgdg12+2)
   pgvector 0.8.7
   distance: 0.3999999761581421
   ```

   `check_setup.py` assumes Ollama on port 18211. With a native Ollama (Option B), change `OLLAMA` at the top of the script to `http://localhost:11434`, or skip to the next step; the rest of the code reads `OLLAMA_URL`.

5. **Index the policies.** `ingest.py` creates the tables, chunks the 30 documents by heading, removes paragraphs that look like instructions to an AI, embeds the chunks and stores them:

   ```bash
   python ingest.py
   ```

   Expected output (the time depends on your CPU):

   ```text
   indexed in 27.3 s: {'added': 30, 'changed': 0, 'unchanged': 0, 'deleted': 0, 'chunks_embedded': 63, 'quarantined': 1}
   30 documents, 63 chunks
   ```

   The one quarantined paragraph is the hostile line planted in `policies/remote-work-coworking.md`. Run `python ingest.py` again after you edit, add or delete a policy: it re-embeds only what changed. `python ingest.py --reset` rebuilds everything.

6. **Ask a question.** The first run downloads the reranker model (about 80 MB, `Xenova/ms-marco-MiniLM-L-6-v2`) into `.models/`, so it needs internet access once.

   ```bash
   python capstone.py "Can I carry unused leave into next year?"
   ```

   Expected output (model answers can vary a little between machines):

   ```text
   Yes, you may carry up to 5 unused days into the next year [1].

   Sources:
     [1] Annual Leave, version 2026.1 (Carrying days over)
     [2] Annual Leave, version 2026.1 (Leaving the company)
     [3] Unpaid Leave and Sabbaticals, version 2024.2 (Unpaid Leave and Sabbaticals)
     [4] Business Travel FAQ, version 2026.1 (Can I extend a trip for a holiday?)

   check: {'refused': False, 'cited': ['leave-annual'], 'invalid': [], 'uncited': False}
   retrieve 653 ms, answer 8.1 s
   ```

   Ask as a manager to see access control at work: `python capstone.py "How many discretionary days can I give someone in my team?" --role managers` cites the Manager Guide to Leave Approvals and answers 2 days; asked without `--role managers`, the manager guide never reaches the prompt.

7. **Run the evaluation.** The retrieval evaluation scores five retrieval setups on the 30 answerable questions in `questions.json`, the labelled evaluation set:

   ```bash
   python l11_eval.py
   ```

   Expected output starts with:

   ```text
   30 labelled questions

   setup                recall@1  recall@3  recall@5    MRR
   vector, no filters      0.817     0.950     0.967  0.921
   vector                  0.883     0.950     0.967  0.956
   keyword                 0.817     0.950     0.950  0.917
   hybrid (RRF)            0.850     0.983     1.000  0.950
   hybrid + rerank         0.917     1.000     1.000  0.983
   ```

   Then the capstone's evidence:

   - `python capstone_scorecard.py` asks all 31 labelled questions plus four unanswerable ones through the finished assistant and ends with the totals the lesson shows (`expected fact in answer 28/30, cites a relevant document 30/30`, `refused 4/5`, `answers asking for a secret: 0   manager chunks shown to employees: 0`). It calls the model for most questions, so on a laptop CPU it takes around ten minutes.
   - `python capstone_gate.py` prints the top reranker score for every question, which is how the answerability threshold of -4 was chosen.
   - `python l10_leaktest.py` checks that no employee question retrieves a manager-only chunk and fails with an `AssertionError` if one does.

8. **Look at the dashboard numbers.** `l16_monitor.py` sends ten realistic questions through the assistant, writes one JSON line per request to `l16_requests.jsonl`, and prints the numbers a dashboard shows: latency p50 and p95, refusals, uncited answers, invalid citations, secret requests, low reranker scores and token counts:

   ```bash
   python l16_monitor.py
   ```

9. **Stop or remove the containers.** `docker stop rag-ollama rag-pg` stops them and `docker start rag-ollama rag-pg` starts them again with the models and data kept. When you are done for good: `docker rm -f rag-ollama rag-pg`, `docker network rm rag-net` and `docker volume rm rag-ollama-models rag-pgdata`.

The scripts each lesson runs on the way are in `lessons/`; [lessons/README.md](lessons/README.md) lists them and how to run them.

## Running the tests

The offline tests need neither Ollama nor PostgreSQL. They check the 30 documents and their metadata, chunking, the injection scanner, the secret-request check, citation checking, the prompt budget, keyword queries, reciprocal rank fusion, the retrieval metrics, and the answerability gate (with stand-ins for the models):

```bash
python -m unittest discover tests -v
```

The last lines are (the time varies):

```text
Ran 17 tests in 0.009s

OK
```

With the full stack running and the index built, `python l10_leaktest.py` is the access-control test the release checklist asks for in continuous integration.

## Project layout

```text
policy-assistant/
  check_setup.py           checks Ollama, PostgreSQL and pgvector (lesson 01)
  ingest.py                create the tables and index the policies incrementally, with the injection scanner
  capstone.py              the command-line Policy Assistant (lesson 17)
  capstone_scorecard.py    every labelled question through the final assistant, checked in code (lesson 17)
  capstone_gate.py         top reranker scores, used to choose the answerability threshold (lesson 17)
  l10_leaktest.py          no employee question may retrieve a manager-only chunk (lesson 10)
  l11_eval.py              recall@k and MRR for five retrieval setups (lesson 11)
  l16_monitor.py           one log line per request, then the dashboard numbers (lesson 16)
  questions.json           the labelled evaluation set: 31 questions with relevant documents and expected facts
  requirements.txt         Python packages
  .env.example             the settings config.py reads, with their defaults
  policies/                the 30 fictional policy documents, markdown with a metadata header
  policy_assistant/
    config.py              URLs, model names and the policy folder, overridable by environment variables
    providers.py           embedding and chat clients for Ollama and OpenAI-compatible APIs (lesson 03)
    documents.py           load documents and their metadata (lesson 04)
    chunking.py            fixed-size, sentence and by-heading chunkers (lesson 04)
    db.py                  schema, HNSW and GIN indexes, connection (lesson 05)
    indexer.py             incremental, transactional indexing (lessons 05, 13 and 14)
    search.py              vector, keyword and hybrid search with filters (lessons 06, 08 and 10)
    rerank.py              cross-encoder reranker through fastembed (lesson 09)
    prompt.py              numbered sources, the guard rule and citation checks (lessons 07 and 13)
    evaluation.py          retrieval metrics (lesson 11)
    guards.py              injection scanner and output check (lesson 13)
    assistant.py           the finished pipeline with the answerability gate (lesson 17)
  lessons/                 the scripts each lesson runs, with an index
  tests/test_offline.py    tests that need no model and no database
```

## Troubleshooting

- **A script hangs with no output (often on Windows).** PostgreSQL is not reachable, and psycopg keeps waiting instead of failing. Check that `rag-pg` is running with `docker ps`, and start it with `docker start rag-pg`. To get an error instead of a wait, add a timeout to the connection string: `export PG_DSN="postgresql://rag:ragpass@localhost:18232/policies?connect_timeout=10"`.
- **`httpx.ConnectError` or "actively refused".** Ollama is not running, or it runs on a different port from `OLLAMA_URL`. Run `docker start rag-ollama` (Option A), or start Ollama and set `OLLAMA_URL=http://localhost:11434` (Option B).
- **`404 Not Found` from `/api/embed` or `/api/chat`, or "model not found".** A model is missing. Pull it with `docker exec rag-ollama ollama pull nomic-embed-text` and `docker exec rag-ollama ollama pull qwen2.5:3b` (or `ollama pull ...`), and check `ollama list`.
- **`port is already allocated` when starting a container.** Something already uses port 18211 or 18232. Pick another host port in `-p` (for example `-p 18233:5432`) and set `PG_DSN` or `OLLAMA_URL` to match.
- **`extension "vector" is not available`.** The database container is a plain PostgreSQL image. Remove it with `docker rm -f rag-pg` and start it again with the `pgvector/pgvector:pg18` image as in step 3.

## License

Copyright (C) 2026 SypherNext

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). You may
use, study and change it. If you share a modified version, or run one as a website or service
that other people use, you must make your full modified source code available under the same
license. See [LICENSE](LICENSE) for the full terms.
