<p align="center">
  <img src="docs/gemini_flash_lite_agent_banner.jpeg" alt="Always-On Agent Memory Layer" width="100%">
</p>

# Always-On Memory Agent v2

**A persistent AI memory system with librarian-grade code search — built with [PydanticAI](https://ai.pydantic.dev/) + Gemini**

Most AI agents have amnesia. They process information, then forget everything. This agent gives any LLM-based system a persistent, evolving memory that runs 24/7 — continuously ingesting, consolidating, decaying, and serving knowledge.

**v2** adds vector search over source code (Librarian mode), smart re-consolidation via a stronger model, rate-limit resilience, and a clean PydanticAI architecture.

## Features

| Feature | Description |
|---|---|
| **Memory Ingestion** | Process text files from an inbox folder or via HTTP API |
| **Consolidation** | Periodic synthesis of memories — finds patterns, connections, and contradictions |
| **Composite Scoring** | Ranks memories by `importance × recency` with explainable recall traces |
| **Memory Decay** | Activity-aware decay loop culls stale, unconsolidated noise |
| **Temporal Truth Windows** | Tracks `valid_from` / `valid_to` to handle evolving facts |
| **Memory Reinforcement** | Consolidation agent boosts importance of confirmed knowledge |
| **Librarian Mode** | Indexes source code via `sqlite-vec` embeddings for semantic file search |
| **Deep Re-Consolidation** | 24h cycle using a smarter model (Gemini 3.0 Flash) for deep insight |
| **Rate-Limit Resilience** | Exponential backoff with retry on 429 / quota errors |
| **Clean Shutdown** | Signal-based (`Ctrl+C` / `SIGTERM`) graceful shutdown |

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        Memory Agent v2                           │
│                                                                  │
│  ┌─────────────┐  ┌──────────────────┐  ┌────────────────────┐  │
│  │ IngestAgent  │  │ ConsolidateAgent │  │    QueryAgent      │  │
│  │ (Flash-Lite) │  │   (Flash-Lite)   │  │   (Flash-Lite)     │  │
│  │              │  │                  │  │                    │  │
│  │ • store      │  │ • read uncons.   │  │ • read memories    │  │
│  │   memory     │  │ • consolidate    │  │ • read history     │  │
│  │              │  │ • reinforce      │  │ • search documents │  │
│  │              │  │ • close truths   │  │ • synthesize       │  │
│  └──────┬───────┘  └────────┬─────────┘  └────────┬───────────┘  │
│         │                   │                      │              │
│  ┌──────┴───────────────────┴──────────────────────┴───────────┐  │
│  │                    SQLite + sqlite-vec                       │  │
│  │  memories │ consolidations │ documents │ vec_documents       │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Background Loops:                                               │
│  • Inbox Watcher (5s)        • Decay Loop (activity-aware)       │
│  • Consolidation (30m)       • Document Indexer (60m)            │
│  • Deep Re-Consolidation (24h, Gemini 3.0 Flash)                 │
└──────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install

```bash
cd local_tools/always-on-memory
pip install -r requirements.txt
```

### 2. Configure

Create a `.env` file (the agent loads it automatically):

```env
GOOGLE_API_KEY="your-gemini-api-key"

MODEL="google-gla:gemini-3.1-flash-lite"
SMART_MODEL="google-gla:gemini-3.0-flash"
EMBEDDING_MODEL="gemini-embedding-2-preview"
RATE_LIMIT="15"

# Comma-separated folders to index for vector search (Librarian mode)
WATCH_DIRS=/home/you/project/src,/home/you/project/docs

# Optional: extra directory names to skip during indexing
IGNORE_DIRS=legacy_code,vendor_lib
```

Get your API key from [Google AI Studio](https://aistudio.google.com/).

### 3. Start

```bash
python agent.py
```

The agent is now:
- 👁️ Watching `./inbox/` for new files
- 🔄 Consolidating every 30 minutes
- 🍂 Decaying stale memories (activity-aware)
- 📚 Indexing source code from `WATCH_DIRS`
- 🧠 Running deep re-consolidation every 24 hours
- 🌐 Serving queries at `http://localhost:8888`

Press `Ctrl+C` to stop cleanly.

### 4. Feed it information

**Option A: Drop files in the inbox**
```bash
echo "JWT auth is preferred over sessions" > inbox/decision.txt
cp architecture_notes.md inbox/
# Agent auto-ingests within 5-10 seconds
```

**Option B: HTTP API**
```bash
curl -X POST http://localhost:8888/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "We use PydanticAI for orchestration", "source": "decision:framework"}'
```

### 5. Query

```bash
curl "http://localhost:8888/query?q=what+architectural+decisions+have+been+made"
```

The query agent will search memories **and** indexed source code, returning an answer with:
- Memory citations: `[Memory 1]`, `[Memory 2]`
- A **Relevant Files** section with paths to matching source files

### 6. Manual triggers

```bash
# Force consolidation now
curl -X POST http://localhost:8888/consolidate

# Force deep re-consolidation (uses smarter model)
curl -X POST http://localhost:8888/reconsolidate
```

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/status` | GET | Memory & index statistics |
| `/memories` | GET | List all stored memories (ranked by composite score) |
| `/ingest` | POST | Ingest text `{"text": "...", "source": "..."}` |
| `/query?q=...` | GET | Query memories + search indexed documents |
| `/consolidate` | POST | Trigger manual consolidation |
| `/reconsolidate` | POST | Trigger deep re-consolidation (smart model) |
| `/delete` | POST | Delete a memory `{"memory_id": 1}` |
| `/clear` | POST | Delete all memories (full reset) |

## CLI Options

```bash
python agent.py [options]

  --watch DIR              Folder to watch for inbox files (default: ./inbox)
  --port PORT              HTTP API port (default: 8888)
  --consolidate-every MIN  Consolidation interval in minutes (default: 30)
```

## Memory Sectors

Memories are categorized into psychological sectors for richer retrieval:

| Sector | Purpose | Examples |
|---|---|---|
| **Semantic** | Facts, rules, architecture decisions | "We use JWT not sessions" |
| **Episodic** | Events, task outcomes, errors | "Login fix deployed on 3/24" |
| **Procedural** | How-to guides, workflows | "Testing pattern for auth module" |
| **Reflection** | Failed reasoning, lessons learned | "Tried X, failed because Y" |

## Librarian Mode (Vector Search)

When `WATCH_DIRS` is set, the agent periodically crawls those directories and indexes source code using `sqlite-vec` embeddings (`gemini-embedding-2-preview`).

**What gets indexed:** `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.md`, `.json`, `.yaml`, `.toml`, and 20+ other code/config extensions.

**What gets skipped (3 layers):**
1. **Binary extension blacklist** — `.pyc`, `.dll`, `.bin`, `.so`, `.exe`, `.wasm`, `.zip`, images, audio, video, fonts, `.lock`, `.min.js`
2. **Directory skip list** — `__pycache__`, `node_modules`, `.git`, `venv`, `dist`, `build`, `.mypy_cache`, `coverage`, `vendor`, `target`, plus anything in `IGNORE_DIRS`
3. **Binary content heuristic** — Reads first 8KB; skips files with >10% non-text bytes

Files are chunked (1500 chars), embedded, and stored in a `vec0` virtual table for cosine similarity search.

## How Decay Works

The decay loop runs periodically and prevents memory bloat:

1. **Activity check** — If no new memories in the last 2 hours, decay is **paused** (prevents amnesia during idle periods)
2. **Age filter** — Only unconsolidated memories older than 24h are candidates
3. **Gradual discount** — Importance reduced by 0.05 per cycle
4. **Culling** — Memories below 0.1 importance are deleted
5. **Consolidated memories are safe** — Only unconsolidated noise decays

## Rate Limit Handling

All LLM calls use `retry_with_backoff()`:
- Detects `429`, `RESOURCE_EXHAUSTED`, and quota errors
- Retries up to 5 times: 30s → 60s → 120s → 240s → 480s
- Non-rate-limit errors propagate immediately

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | (required) | Gemini API key |
| `MODEL` | `google-gla:gemini-3.1-flash-lite` | Lite model for ingest/consolidate/query |
| `SMART_MODEL` | `google-gla:gemini-3.0-flash` | Smart model for deep re-consolidation |
| `EMBEDDING_MODEL` | `gemini-embedding-2-preview` | Model for vector embeddings |
| `MEMORY_DB` | `memory.db` | SQLite database path |
| `RATE_LIMIT` | `15` | Max concurrent model requests |
| `WATCH_DIRS` | (empty) | Comma-separated dirs for Librarian mode |
| `IGNORE_DIRS` | (empty) | Extra directory names to skip |

## Project Structure

```
always-on-memory/
├── agent.py          # Main agent (PydanticAI + aiohttp)
├── requirements.txt  # Dependencies
├── .env              # Configuration (auto-loaded)
├── inbox/            # Drop files here for auto-ingestion
├── memory.db         # SQLite + sqlite-vec database (auto-created)
└── docs/             # Assets
```

## Built With

- [PydanticAI](https://ai.pydantic.dev/) — Agent framework with typed tools and structured output
- [Gemini 3.1 Flash-Lite](https://ai.google.dev/) — Fast, cheap LLM for continuous background operation
- [Gemini 3.0 Flash](https://ai.google.dev/) — Smarter model for deep re-consolidation
- [sqlite-vec](https://github.com/asg017/sqlite-vec) — Vector search extension for SQLite
- [aiohttp](https://docs.aiohttp.org/) — Async HTTP server

## Credits

This project was derived from the [Google Cloud Platform Always-On-Memory Agent](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/gemini/agents/always-on-memory-agent) and has been expanded with PydanticAI, Librarian mode (vector search), and rate-limiting resilience.

## License

MIT
