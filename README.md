# Customer Support RAG Agent

LangGraph-based customer support bot that answers only from ingested policy documents, grades retrieved chunks before generation, and blocks ungrounded answers with a faithfulness check.

This rewrite keeps the original project idea intact, but fixes the issues that would break a fresh local run.

## What this project does

- Ingests policy documents from `.docx`, `.pdf`, or `.html`
- Chunks them with structure awareness so tables are not split incorrectly
- Stores embeddings in FAISS
- Retrieves the most relevant chunks for a support question
- Grades retrieved chunks before generation
- Runs a faithfulness check before returning the final answer
- Falls back to a safe support message when context is missing or ungrounded

## Project structure

```text
customer-support-rag-agent/
├── customer_support_bot/
│   ├── agent/
│   │   ├── cache.py
│   │   ├── graph.py
│   │   ├── nodes.py
│   │   └── state.py
│   ├── api/
│   │   └── app.py
│   ├── ingestion/
│   │   ├── chunker.py
│   │   ├── document_loader.py
│   │   ├── embedder.py
│   │   └── ingest.py
│   ├── __init__.py
│   ├── config.py
│   └── prompts.py
├── sample_docs/
│   └── acme_sample_policy.docx
├── main.py
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── README.md
```

## Setup

```bash
git clone https://github.com/imbhaskarm/customer-support-rag-agent.git
cd customer-support-rag-agent
python -m venv .venv
```

Activate the environment:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create `.env`:

```bash
cp .env.example .env
```

Add your Groq key to `.env`.

## First run

A fresh clone does **not** contain a built FAISS index. Build it first:

```bash
python -m customer_support_bot.ingestion.ingest --tenant-id acme --filepath sample_docs/acme_sample_policy.docx
```

Then run the CLI:

```bash
python main.py
```

Or run the API:

```bash
uvicorn customer_support_bot.api.app:app --reload
```

## Example flow

1. Ingest the sample policy document into `faiss_indexes/tenant_acme`
2. Ask a question like: `Can I get a refund after activating a digital product?`
3. The graph retrieves relevant chunks, grades them, generates an answer, and validates faithfulness
4. If grounding is weak, the bot returns a safe fallback instead of guessing

## Bugs fixed from the original repo

| Bug | Fix |
|---|---|
| `ChatGroq(api_key=...)` used the wrong keyword | Fixed to `groq_api_key=...` |
| No runnable ingestion entry point | Added `customer_support_bot/ingestion/ingest.py` |
| First clone crashed because no FAISS index existed | Added explicit first-run ingestion step and clear error message |
| Redis connection attempted at import time | Switched to lazy Redis initialization |
| `langgraph` graph used old entry setup style | Rebuilt with `START` and compiled graph flow |
| Evaluation packages were mixed into runtime deps | Split into `requirements.txt` and `requirements-dev.txt` |
| No simple local test path | Added `main.py` CLI runner |
| Repo had no clean Python package layout | Reorganized into `customer_support_bot/` package |

## Working condition

This project is designed to work on a normal local machine after these two commands:

```bash
python -m customer_support_bot.ingestion.ingest --tenant-id acme --filepath sample_docs/acme_sample_policy.docx
python main.py
```

If Redis is not running locally, the bot still works. Cache and session memory are skipped gracefully.

## GitHub positioning

**Repo description**

LangGraph customer support RAG agent with FAISS retrieval, chunk grading, and faithfulness checks.

**Topics**

- langgraph
- rag
- customer-support
- faiss
- groq

**Resume bullets**

- Built a customer support RAG agent that retrieves policy content from FAISS and uses LangGraph to control retrieval, grading, generation, and fallback flow.
- Implemented structure-aware ingestion for DOCX, PDF, and HTML documents to preserve headings, sections, and tables during chunking.
- Added a faithfulness gate and safe fallback response to reduce hallucinations in customer-facing support workflows.

**LinkedIn-ready summary**

Built a customer support bot in Python using LangGraph, FAISS, Groq, and HuggingFace embeddings. The project ingests policy documents, retrieves relevant sections, grades chunk relevance, and only returns answers that pass a grounding check.

**Pin on GitHub?**

Yes. This is a strong learning project because it shows more than basic chat or simple RAG.
