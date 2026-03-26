<p align="center">
  <img src="docs/gemini_flash_lite_agent_banner.jpeg" alt="Always-On Agent Memory Layer" width="100%">
</p>

# Always-On Memory Agent v3

**A self-optimizing, portable memory system — built with [PydanticAI](https://ai.pydantic.dev/) + Gemini**

Most AI agents have amnesia. They process information, then forget everything. This agent gives any LLM-based system a persistent, evolving memory that runs 24/7 — continuously ingesting, consolidating, dreaming, and serving knowledge.

**v3** transforms Always-On-Memory into a self-optimizing system, introducing portable "MemCubes," adversarial consolidation harnesses, active AutoDream optimization cycles, and TurboQuant-inspired vector compression.

## Features

| Feature | Description |
|---|---|
| **MemCubes** | Standardized, portable memory units for cross-platform migration |
| **Adversarial Consolidation** | Multi-agent (Generator-Evaluator) harness for high-fidelity memory synthesis |
| **AutoDream Cycles** | Idle-time "sleep" phase that prunes, reorganizes, and clusters MemCubes |
| **TurboQuant 3.5-bit** | Random orthogonal rotations + scalar quantization (3.5-bit precision) for high-fidelity vector compression |
| **Memory Ingestion** | Multimodal ingestion (text, images, audio, video) via Inbox or HTTP |
| **Librarian Mode** | High-performance semantic code search with debounced indexing |
| **Deep Re-Consolidation** | 24h quality audit using the smartest available models (Gemini 2.0 Pro) |
| **Self-Improvement** | Autonomously evolves project-specific skills and patterns (EvoSkill) |
| **Rate-Limit Resilience** | Production-grade exponential backoff for Gemini API quotas |

## Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                        Always-On-Memory (v3)                                             │
│                                                                                          │
│  ┌──────────────────┐  ┌────────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │   agent.py       │  │ agents_factory.py  │  │   librarian.py   │  │    server.py    │ │
│  │                  │  │                    │  │                  │  │                 │ │
│  │ • Process Leader │  │ • Gen-Eval Harness │  │ • TurboQuant     │  │ • REST API      │ │
│  │ • AutoDream Loop │  │ • PydanticAI       │  │ • Vector Indexer │  │ • Cube Im/Ex    │ │
│  │ • Inbox Watcher  │  │ • Tool Mapping     │  │ • Search Logic   │  │ • Search/Query  │ │
│  └────────┬─────────┘  └────────┬───────────┘  └────────┬─────────┘  └──────┬──────────┘ │
│           │                     │                       │                   │            │
│  ┌────────┴─────────────────────┴───────────────────────┴───────────────────┴──────────┐ │
│  │                                 Shared Layer                                        │ │
│  │                    (config.py │ models.py │ database.py │ utils.py)                 │ │
│  └────────┬─────────────────────┬───────────────────────┬───────────────────┬──────────┘ │
│           │                     │                       │                   │            │
│  ┌────────┴─────────────────────┴───────────────────────┴───────────────────┴──────────┐ │
│  │                            MemCube Store (SQLite-Vec)                               │ │
│  │             memories (int8) │ consolidations │ files │ vec_index (int8)             │ │
│  └─────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  Active Cycles:                                                                          │
│  • Inbox Watcher (5s)        • AutoDream (When Idle > 30m)                               │
│  • Consolidation (30m)       • Librarian Indexer (Debounced 60s)                         │
│  • Deep Re-Consolidation (24h)                                                           │
│  • Self-Improvement Audit (24h)                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Install

```bash
git clone https://github.com/A4ABATTERY/Always-On-Memory.git
cd Always-On-Memory
python3 -m venv .venv
source .venv/bin/activate  # On Linux/macOS
pip install -r requirements.txt
```

### 2. Configure

Create a `.env` file (the agent loads it automatically):

```env
GOOGLE_API_KEY="your-gemini-api-key"

MODEL="gemini-3.1-flash-lite"
SMART_MODEL="gemini-3.0-flash"
EMBEDDING_MODEL="gemini-embedding-2-preview"
RATE_LIMIT="15"

# Comma-separated folders to index for vector search (Librarian mode)
WATCH_DIRS=/home/you/project/src,/home/you/project/docs

# Extra directory names to skip during indexing
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

### 6. Verify with Tests

Run the unit test suite to ensure everything is working correctly:

```bash
PYTHONPATH=. ./.venv/bin/python -m unittest discover tests
```

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/status` | GET | MemCube & index statistics |
| `/memories` | GET | List all stored MemCubes (ranked by score) |
| `/ingest` | POST | Ingest text or metadata `{"text": "...", "source": "..."}` |
| `/query?q=...` | GET | Query MemCubes + search documents (synthesized answer) |
| `/search?q=...` | GET | Direct semantic search over files (snippets + paths) |
| `/export_cubes` | GET | Export MemCubes as portable JSON `?ids=1,2,3` |
| `/import_cubes` | POST | Import portable MemCube JSON `{"cubes": [...]}` |
| `/consolidate` | POST | Trigger manual adversarial consolidation |
| `/reconsolidate` | POST | Trigger deep re-consolidation (Pro model) |
| `/improve` | POST | Trigger manual self-improvement audit |
| `/delete` | POST | Delete a MemCube `{"memory_id": 1}` |
| `/clear` | POST | Full MemOS reset (clears database & inbox) |

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

When `WATCH_DIRS` is set, the agent periodically scans those directories for changes using `os.path.getmtime`. To avoid indexing while files are still being actively modified (e.g., by an LLM), it implements a **60-second debounce timer**.

- **Detection**: Checks for modifications every 5 seconds (`SCAN_INTERVAL`).
- **Debounce**: Waits for a 60-second quiet window (`DEBOUNCE_INTERVAL`) after the last detected change before indexing.
- **Indexing**: Uses MD5 hashing to confirm changes before generating `sqlite-vec` embeddings (`gemini-embedding-2-preview`).

**What gets indexed:** `.py`, `.js`, `.ts`, `.go`, `.rs`, `.java`, `.md`, `.json`, `.yaml`, `.toml`, and 20+ other code/config extensions.

**What gets skipped (3 layers):**
1. **Binary extension blacklist** — `.pyc`, `.dll`, `.bin`, `.so`, `.exe`, `.wasm`, `.zip`, images, audio, video, fonts, `.lock`, `.min.js`
2. **Directory skip list** — `__pycache__`, `node_modules`, `.git`, `venv`, `dist`, `build`, `.mypy_cache`, `coverage`, `vendor`, `target`, plus anything in `IGNORE_DIRS`
3. **Binary content heuristic** — Reads first 8KB; skips files with >10% non-text bytes

Files are chunked (1500 chars), embedded, and stored in a `vec0` virtual table for cosine similarity search.

## How AutoDream Works

The AutoDream cycle replaces passive decay with active optimization using embedding-based clustering:

1. **Idle Detection** — If no system activity occurs for 30 minutes, the Dream sequence begins.
2. **Importance Decay** — Gradual discount of importance scores for unconsolidated memories (-0.05).
3. **Redundancy Pruning** — Identifies and deletes exact or near-duplicate MemCubes.
4. **Embedding Clustering** — Groups related MemCubes using greedy semantic similarity (numpy-optimized).
5. **Adversarial Synthesis** — For clusters >= 3, triggers the Generator-Evaluator harness to compress the cluster into a single, high-fidelity Insight cube.
6. **Zero Bloat** — Ensures the system remains streamlined without manual maintenance.

## Rate Limit Handling

All LLM calls use `retry_with_backoff()`:
- Detects `429`, `RESOURCE_EXHAUSTED`, and quota errors
- Retries up to 5 times: 30s → 60s → 120s → 240s → 480s
- Non-rate-limit errors propagate immediately

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | (required) | Gemini API key |
| `MODEL` | `gemini-3.1-flash-lite` | Lite model for ingest/consolidate/query |
| `SMART_MODEL` | `gemini-3.0-flash` | Smart model for deep re-consolidation |
| `EMBEDDING_MODEL` | `gemini-embedding-2-preview` | Model for vector embeddings |
| `MEMORY_DB` | `memory.db` | SQLite database path |
| `RATE_LIMIT` | `15` | Max concurrent model requests |
| `WATCH_DIRS` | (empty) | Comma-separated dirs for Librarian mode |
| `IGNORE_DIRS` | (empty) | Extra directory names to skip |
| `SKILLS_DIR` | `.agent/skills` | Directory where skills are stored |

## Project Structure

The project has been refactored into focused, type-safe modules:

- `agent.py`: Principal entry point and background loop orchestrator.
- `agents_factory.py`: PydanticAI agent definitions and tool mapping.
- `config.py`: Centralized environment variable loading and settings.
- `database.py`: SQLite connection and `sqlite-vec` initialization.
- `librarian.py`: Semantic file search (Librarian) logic and debounce loop.
- `memory_store.py`: CRUD operations for memory persistence and rankings.
- `models.py`: Immutable Pydantic data models for core entities.
- `server.py`: aiohttp API server implementation.
- `utils.py`: Shared utilities for embeddings and error handling.
- `tests/`: Comprehensive unit test suite covering core functionality.

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
