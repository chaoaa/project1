# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start Qdrant (vector DB, required before backend)
docker compose up -d qdrant

# Backend (uvicorn with auto-reload)
cd backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (Streamlit)
cd frontend && streamlit run streamlit_app.py

# Install backend dependencies
cd backend && pip install -r requirements.txt

# Install frontend dependencies
cd frontend && pip install -r requirements.txt

# Docker full stack (backend + frontend + qdrant)
docker compose --profile full up -d
```

## Architecture

### Three-tier system
- **FastAPI backend** (`backend/app/`) — REST API at `:8000`
- **Streamlit frontend** (`frontend/streamlit_app.py`) — Web UI at `:8501`
- **Qdrant** — Vector database (start via `docker compose up -d qdrant`)

### Document ingestion pipeline
`/api/documents/upload` → `document_loader.py` (multi-format parse) → `text_cleaner.py` (noise removal) → `text_splitter.py` (hierarchical chunking: sections → paragraphs → sentences) → `embedding_service.py` (sentence-transformers) → `vector_store.py` (Qdrant upsert) → `bm25_store.py` (rebuild BM25 index)

### LangGraph Agent (graph.py)
8-node state machine with 5 intents:
1. `classify_intent_node` — LLM classifies question into intent
2. `retrieve_node` — hybrid search (skipped for version_compare)
3. `route_tool_node` — conditional routing based on intent
4. `{policy_qa | eligibility_check | checklist_generation | version_compare}_node` — tool execution
5. `final_answer_node` — logging/reporting

Conditional edges: `classify_intent → retrieve → route_tool → [4 tool nodes] → final_answer → END`

### Hybrid search (retriever.py)
Qdrant vector search + rank-bm25 keyword search → min-max normalization per channel → weighted fusion (0.65 vector + 0.35 BM25) → dedup by chunk_id → global ranking

### Confidence & refusal (citation_service.py)
Three tiers based on top-1 fusion score: high (≥0.75) → answer, medium (≥0.45) → answer with caveat, low (<0.45) → refuse. Applies before LLM call.

### Backend services layer (`backend/app/services/`)
| Service | Role |
|---|---|
| `document_loader.py` | Parse PDF/DOCX/TXT/MD/HTML |
| `text_cleaner.py` | Remove page numbers, HTML tags, noise; keep section headers |
| `text_splitter.py` | Hierarchical chunking with configurable size/overlap |
| `embedding_service.py` | sentence-transformers (default: `BAAI/bge-m3`), singleton with lazy load |
| `vector_store.py` | Qdrant CRUD (collection, upsert, search, scroll) |
| `bm25_store.py` | rank-bm25 + jieba, in-memory, rebuilt on each upload |
| `retriever.py` | Hybrid search orchestration + score fusion |
| `llm_client.py` | OpenAI-compatible client with JSON fallback; swap by changing `.env` |
| `citation_service.py` | Citation formatting + confidence assessment |

### Agent layer (`backend/app/agent/`)
| File | Role |
|---|---|
| `state.py` | `AgentState` TypedDict (question, intent, retrieved_chunks, answer, etc.) |
| `graph.py` | `StateGraph` definition with 8 nodes + conditional edges |
| `tools.py` | 4 tools: `search_policy`, `check_eligibility`, `generate_checklist`, `compare_policy_versions` |
| `prompts.py` | All prompt templates (intent classification, RAG, eligibility, checklist, version compare) |

### API routes
- `GET /api/health` — Health check (Qdrant connectivity, document/chunk counts)
- `POST /api/documents/upload` — Upload + parse + embed + index
- `GET /api/documents` — List indexed documents
- `POST /api/chat/query` — Simple RAG (no agent)
- `POST /api/chat/agent` — LangGraph agent (intent classification → tool calling)
- `POST /api/chat/retrieve` — Debug: hybrid search results only

### Key design decisions
- **Config**: `pydantic-settings` reads from `.env` in project root; restart backend to apply changes
- **Storage**: `metadata.json` for document index, Qdrant for chunk vectors+text, disk for raw uploads
- **BM25**: Rebuilt from Qdrant on startup (`lifespan` hook) and after each upload
- **LLM compatibility**: OpenAI SDK with configurable `base_url`/`model`/`api_key`; `structured_chat` falls back to regex JSON extraction if `response_format` unsupported
- **100+ comments in code** explaining design rationale per module
