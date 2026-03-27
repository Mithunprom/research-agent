# Research Agent

An adaptive research assistant built with LangGraph that answers complex questions by combining local RAG retrieval (FAISS) with web search, self-evaluates answers for groundedness, and iteratively refines until quality thresholds are met.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Graph Flow](#graph-flow)
- [Source Files](#source-files)
- [Data & Persistence](#data--persistence)
- [Key Design Decisions](#key-design-decisions)
- [Dependencies](#dependencies)

---

## Overview

The agent accepts a natural-language research question and runs it through an 11-node LangGraph pipeline:

1. **Plan** - An LLM planner decides the retrieval strategy (local-only, hybrid, or web-first) and whether the question needs clarification.
2. **Retrieve** - Fetches relevant chunks from a local FAISS index and/or the web via Tavily, then reranks with a cross-encoder.
3. **Answer** - An LLM generates a structured, cited answer from the merged context.
4. **Judge** - A second LLM pass evaluates groundedness (score 0-1). If the score is below 0.75 and iterations remain, the query is refined and the loop restarts.
5. **Persist** - All artifacts (answer, judge verdict, citations, session notes) are saved to disk.

Multi-turn conversation memory is maintained per session so follow-up questions build on prior context.

---

## Architecture

See the architecture diagram: **`docs/architecture_diagram.md`** (Mermaid) or the FigJam board linked below.

```
User
 │
 ▼
chat_cli.py / run_cli.py
 │
 ▼
┌──────────────────────────────────────────────────────────────────┐
│                     LangGraph Agent Pipeline                     │
│                                                                  │
│  init ──▶ plan ──┬──▶ clarify ──▶ persist ──▶ END               │
│                  │                                               │
│                  └──▶ retrieve_local ──┬──▶ web_search ──┐      │
│                                       │                  │      │
│                                       └──▶ compose ◀─────┘      │
│                                              │                   │
│                                              ▼                   │
│                                           answer                 │
│                                              │                   │
│                                              ▼                   │
│                                           judge                  │
│                                           │    │                 │
│                                    refine ◀    ▶ write_notes     │
│                                      │              │            │
│                                      ▼              ▼            │
│                                    plan          persist ──▶ END │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
 │                    │                     │
 ▼                    ▼                     ▼
FAISS Index      Tavily API           Session Memory
(data/faiss_index)                   (data/sessions/)
                                     (data/runs/)
```

---

## Project Structure

```
research_agent/
├── .env                          # API keys (OPENAI_API_KEY, TAVILY_API_KEY)
├── requirements.txt              # Python dependencies
├── PROJECT.md                    # This file
│
├── docs/                         # Source documents for RAG ingestion
│   ├── Designing Machine Learning Systems.pdf
│   └── Practical MLOps.pdf
│
├── src/                          # All source code
│   ├── agent_graph.py            # Core LangGraph pipeline (11 nodes)
│   ├── chat_cli.py               # Interactive multi-turn REPL
│   ├── run_cli.py                # Single-query CLI
│   ├── rag_ingest_local.py       # PDF/text ingestion → FAISS index
│   ├── tools_web.py              # Tavily search + HTML fetch
│   ├── tools_rerank.py           # Cross-encoder reranking
│   ├── tools_memory.py           # Session persistence (notes, turns)
│   └── RAG_FLOW.md               # Technical RAG documentation
│
├── data/
│   ├── faiss_index/              # Pre-built FAISS vector index
│   ├── runs/<run_id>/            # Per-run artifacts
│   │   ├── answer.md
│   │   ├── judge.json
│   │   └── citations.json
│   └── sessions/<session_id>/    # Per-session memory
│       ├── notes.md              # Accumulated research notes
│       ├── sources.jsonl         # Source references (one JSON per line)
│       └── turns.jsonl           # Q&A turn log
```

---

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set API keys in .env
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...

# 3. Place source documents in docs/

# 4. Build the FAISS index
python -m src.rag_ingest_local
```

---

## Usage

### Interactive mode (multi-turn)

```bash
python -m src.chat_cli
python -m src.chat_cli --session my_topic --max-iters 3
```

- Type questions at the `You:` prompt
- Follow-up questions reuse the same session context
- Type `exit` or `quit` to stop

### Single query

```bash
python -m src.run_cli "What are feedback loops in ML systems?"
```

---

## Graph Flow

```
init → plan →[needs clarification?]
           ├─ YES → clarify → persist → END
           └─ NO  → retrieve_local →[route_retrieval]
                        ├─ web_first     → web_search → compose
                        ├─ local_only    → compose
                        └─ hybrid (<800 chars local) → web_search → compose
                    compose → answer → judge →[loop_or_finish]
                        ├─ PASS (score >= 0.75) or max_iters → write_notes → persist → END
                        └─ NEEDS_MORE_RESEARCH               → refine → plan (re-plan)
```

### Node descriptions

| Node | Purpose |
|------|---------|
| `init` | Set defaults (session_id, run_id, k values), load prior notes and recent turns |
| `plan` | LLM planner decides retrieval_mode, k values, and whether to clarify |
| `clarify` | Return a clarifying question instead of answering |
| `retrieve_local` | Query FAISS (top k_local), rerank to top 5 via cross-encoder |
| `route_retrieval` | Conditional: web_first / local_only / hybrid (fallback if < 800 chars) |
| `web_search` | Tavily API + fetch page text + rerank to top 5 |
| `compose` | Merge local + web sources, deduplicate, number as [1]..[10] |
| `answer` | LLM generates cited answer using context + prior notes + recent turns |
| `judge` | LLM evaluates groundedness → JSON {verdict, score, missing_points, suggested_queries} |
| `loop_or_finish` | Route to refine (if score < 0.75 and iters < max) or finalize |
| `refine` | Replace question with judge's first suggested_query |
| `write_notes` | LLM summarises answer into <= 12 bullet points for session memory |
| `persist` | Save answer.md, judge.json, citations.json; append session notes/sources/turns |

---

## Source Files

### agent_graph.py
Core pipeline. Defines the `State` TypedDict, all 11 node functions, 4 system prompts (ANSWER, JUDGE, NOTES, PLANNER), and the graph wiring. Uses `gpt-4.1-mini` via `langchain_openai`.

### chat_cli.py
Interactive REPL. Parses `--session` and `--max-iters` args, runs an input loop, handles clarification follow-ups, and prints answer/judge/session metadata.

### run_cli.py
Single-shot CLI. Takes a question as a positional argument, invokes the graph once, and prints results.

### rag_ingest_local.py
Ingestion pipeline. Reads PDFs (page-by-page via pypdf) and text files from `docs/`, splits into ~900-char chunks with 120-char overlap, embeds with `all-MiniLM-L6-v2` (384-dim), and builds a FAISS flat index saved to `data/faiss_index/`.

### tools_web.py
- `tavily_search(query, k)` — Tavily API wrapper returning top-K results
- `fetch_text(url, timeout)` — Downloads a page, strips HTML with BeautifulSoup, returns first 5000 chars

### tools_rerank.py
- `rerank(query, candidates, top_n)` — Cross-encoder reranking using `ms-marco-MiniLM-L-6-v2`. Lazy-loads the model as a singleton.

### tools_memory.py
Session persistence layer:
- `default_session_id()` — today's date
- `session_dir()` / `load_session()` / `append_session()` — notes.md + sources.jsonl
- `append_turn()` / `load_recent_turns()` / `format_turns_for_prompt()` — turns.jsonl

---

## Data & Persistence

### Per-run artifacts (`data/runs/<run_id>/`)

| File | Content |
|------|---------|
| `answer.md` | Final answer text with [1], [2] citation markers |
| `judge.json` | `{verdict, score, missing_points, suggested_queries}` |
| `citations.json` | Numbered list of source dicts (id, title, text, url/page, rerank_score) |

### Per-session memory (`data/sessions/<session_id>/`)

| File | Content |
|------|---------|
| `notes.md` | Append-only bullet-point research notes |
| `sources.jsonl` | One JSON source object per line |
| `turns.jsonl` | One JSON turn per line: `{question, answer, sources, judge}` |

Sessions are keyed by date (auto) or by a custom name (`--session my_topic`).

---

## Key Design Decisions

1. **Plan-first retrieval** — The planner LLM decides the retrieval strategy *before* fetching, avoiding wasted web calls for questions the local index can answer and vice versa.

2. **Retrieve-then-rerank** — FAISS retrieves a broad set (k=20), then a cross-encoder narrows to the top 5. This balances recall with precision.

3. **Self-judging loop** — A dedicated judge prompt evaluates groundedness and suggests refinements. The loop is capped at `max_iters` to bound cost and latency.

4. **Append-only session memory** — Notes and sources are never overwritten, only appended. Recent turns (last 6) are injected into prompts for conversational continuity.

5. **Separation of concerns** — Web tools, memory, reranking, and the graph are in separate modules. The graph itself is a pure function (`make_graph()`) returning a compiled runnable.

---

## Dependencies

| Category | Package | Purpose |
|----------|---------|---------|
| LLM | `langchain-openai` | GPT-4.1-mini interface |
| Graph | `langgraph` | Stateful agent pipeline |
| Embeddings | `langchain-huggingface`, `sentence-transformers` | all-MiniLM-L6-v2 (384-dim) |
| Vector store | `faiss-cpu` | Local similarity search |
| Reranking | `sentence-transformers` | ms-marco-MiniLM-L-6-v2 cross-encoder |
| Web search | `tavily-python` | Tavily API |
| HTML parsing | `beautifulsoup4`, `requests` | Page text extraction |
| PDF | `pypdf` | PDF text extraction |
| Config | `python-dotenv` | .env file loading |
| Framework | `langchain`, `langchain-community` | Core abstractions |
