# RAG‑Powered FastAPI Service

A lightweight Retrieval‑Augmented Generation (RAG) API built with **FastAPI**, **LangChain**, and a pluggable LLM backend (OpenAI, Llama.cpp, etc.).

---

## ✨ Features

* **Vector search** via ChromaDB (`db/`)
* **Plug‑and‑play LLMs** – local `llama‑cpp` or cloud (`gpt‑4o`, Anthropic, Cohere…)
* **Hot‑reload dev server** (`--reload`) with VS Code Debug profile
* Simple `/query` endpoint that returns JSON `{"answer": "…"}`

---

## 🔧 Requirements

| Tool                 | Version              | Notes                                 |
| -------------------- | -------------------- | ------------------------------------- |
| Python               | ≥ 3.9                | virtualenv recommended                |
| pip                  | latest               | `python -m pip install --upgrade pip` |
| git                  | any                  | to clone the repo                     |
| (optional) GCC/clang | for llama‑cpp builds |                                       |

> **OpenAI users**: set `OPENAI_API_KEY` in your environment or `.env`.

---

## 🚀 Quick start

```bash
# 1) Clone & enter project
$ git clone <repo-url>
$ cd <project>

# 2) Create & activate venv
$ python -m venv .venv
$ source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3) Install Python deps
$ pip install -r requirements.txt

# 4) Copy sample config if needed
$ cp config.sample.yaml config.yaml  # adjust as required

# 5) (Optional) create .env for secrets
$ echo "OPENAI_API_KEY=sk-..." > .env
```

### Environment variables

| Var                 | Purpose                      | Default                          |
| ------------------- | ---------------------------- | -------------------------------- |
| `CONFIG_PATH`       | Path to `config.yaml`        | `${workspaceFolder}/config.yaml` |
| `VECTOR_STORE_PATH` | Path to Chroma DB            | `${workspaceFolder}/db`          |
| `OPENAI_API_KEY`    | OpenAI key (if using OpenAI) | –                                |

Both `CONFIG_PATH` & `VECTOR_STORE_PATH` are auto‑injected by the VS Code launch profile below.

---

## 🐞 Running with VS Code (recommended for dev)

1. Open the project folder in VS Code.
2. Ensure the following **launch configuration** exists in `.vscode/launch.json` (already present if you cloned the repo):

```jsonc
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "FastAPI: Uvicorn",
      "type": "debugpy",
      "request": "launch",
      "module": "uvicorn",
      "args": [
        "app:app",
        "--host", "127.0.0.1",
        "--port", "8000",
        "--reload",
        "--timeout-keep-alive", "120"
      ],
      "jinja": true,
      "justMyCode": true,
      "env": {
        "CONFIG_PATH": "${workspaceFolder}/config.yaml",
        "VECTOR_STORE_PATH": "${workspaceFolder}/db"
      }
    }
  ]
}
```

3. Hit **F5** (or the green ▶︎) to start the API in debug‑reload mode.

> The debug console will show `Uvicorn running on http://127.0.0.1:8000`.

---

## 🏃‍♀️ Running from the command line

```bash
uvicorn app:app \
  --host 127.0.0.1 \
  --port 8000 \
  --reload \
  --timeout-keep-alive 120
```

Set the same `CONFIG_PATH` / `VECTOR_STORE_PATH` environment vars if you moved them:

```bash
export CONFIG_PATH=$(pwd)/config.yaml
export VECTOR_STORE_PATH=$(pwd)/db
```

---

## 🔌 Switching LLM back‑ends

`config.yaml` controls the active provider:

```yaml
llm_provider: openai_chat        # options: openai_chat, openai, llamacpp, anthropic, cohere, hf
llm_opts:
  model_name: gpt-4o             # e.g. gpt-3.5-turbo, gpt-4o-mini, llama-2-7b.Q4_K_M.gguf
  temperature: 0.0
```

* Local models require a `model_path`.
* Cloud models need the relevant API key (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, …).

---

## 🛠️ Populating the Vector Store

1. Prepare your corpus (PDFs, MD files, etc.).
2. Use the provided `scripts/ingest.py` or your own loader to chunk & embed:

```bash
python scripts/ingest.py --input ./docs --db ./db
```

> The script respects the same `config.yaml` chunking & embedding settings.

---

## 🔌 API usage

* **POST /query** – ask a question and receive an answer.

```bash
curl -X POST http://127.0.0.1:8000/query \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the project license?"}'
```

Sample response:

```json
{
  "answer": "The project is licensed under the MIT License."
}
```

---

## 📝 License

This project is released under the **MIT License**. See `LICENSE` for details.
