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
| **TurboQuant-inspired int8** | Random orthogonal rotations + int8 scalar quantization (~75% storage reduction) for high-fidelity vector compression |
| **Structural Linkage** | Proactive background auditing and autonomous "Self-Healing" of code-memory connections (V3.3) |
| **Memory Ingestion** | Recursive, hash-gated multimodal ingestion (text, images, audio, video) from Inbox |
| **Semantic Invalidation**| Automated hard-expiry (`valid_to`) of superseded memories upon source file updates |
| **Librarian Mode** | High-performance semantic code search with structural chunking |
| **Lexical Symbol Index**| Fast O(log n) identifier lookups (functions, classes, constants) |
| **Deep Re-Consolidation** | 24h quality audit using the smartest available models (Gemini 2.0 Pro) |
| **Self-Improvement** | Autonomously evolves project-specific skills and patterns (EvoSkill) |
| **Rate-Limit Resilience** | Production-grade exponential backoff for Gemini API quotas |

## Architecture

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                        Always-On-Memory (v3)                                             │
│                                                                                          │
│  ┌──────────────────┐  ┌────────────────────┐  ┌──────────────────┐  ┌─────────────────┐ │
│  │  Process Leader  │  │   Agent Factory    │  │     Librarian    │  │   Server (API)  │ │
│  │  (agent.py)      │  │ (agents_factory.py)│  │  (librarian.py)  │  │    (server.py)  │ │
│  ├──────────────────┤  ├────────────────────┤  ├──────────────────┤  ├─────────────────┤ │
│  │ • asyncio.run()  │  │ • PydanticAI       │  │ • Vector Indexer │ • REST API      │ │
│  │ • AutoDream Loop │  │ • Gen-Eval Harness │  │ • os.scandir Walk│ • Cube Im/Ex    │ │
│  │ • Inbox Watcher  │  │ • Tool Mapping     │  │ • Threaded I/O   │ • Search/Query  │ │
│  └────────┬─────────┘  └────────┬───────────┘  └────────┬─────────┘  └──────┬──────────┘ │
│           │                     │                       │                   │            │
│  ┌────────┴─────────────────────┴───────────────────────┴───────────────────┴──────────┐ │
│  │                                 Shared Layer                                        │ │
│  │    (config.py │ models.py │ database.py │ utils.py │ memory_store.py)               │ │
│  └────────┬─────────────────────┬───────────────────────┬───────────────────┬──────────┘ │
│           │                     │                       │                   │            │
│  ┌────────┴─────────────────────┴───────────────────────┴───────────────────┴──────────┐ │
│  │                            MemCube Store (SQLite-Vec)                               │ │
│  │             memories (int8) │ consolidations │ files │ vec_index (int8)             │ │
│  └─────────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                          │
│  Active Cycles:                                                                          │
│  • Recursive Inbox Watcher (5s) • AutoDream (Idle > 30m) • Proactive Sync (Drift Det.)   │
│  • Consolidation (30m)       • Librarian Indexer (60s)   • Self-Healing Audit (V3.3)     │
│  • Deep Re-Consolidation (24h)                                                           │
│  • Self-Improvement Audit (24h)                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

## Features (V3.3 Upgrade)

### 🧠 Structural Memory Linkage
V3.3 introduces a proactive **Self-Healing** layer for code-memory grounding:
1. **Librarian Mode (Vector Search)**: 
    - **High Performance**: Uses optimized recursive walkers (`os.scandir`) to prune ignored branches (e.g. `node_modules`) at the walk start, ensuring near-instant indexing.
    - **Threaded Execution**: Moves FS-heavy modification checks to background threads to prevent event loop blocking on slow file systems (Windows/OneDrive).
    - **Drift Detection**: Compares current embeddings with linked `MemCubes` to detect semantic drift.
2. **Proactive Sync**: If distance exceeds `DRIFT_THRESHOLD` (0.18), the cube is sent to the **Sync Auditor** agent.
3. **Link Evolution**: Link statuses evolve from `ACTIVE` to `HISTORICAL` or `REPAIR` autonomously.

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

MODEL="gemini-3.1-flash-lite-preview"
SMART_MODEL="gemini-3-flash-preview"
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

**Option A: Drop files or folders in the inbox**
```bash
echo "JWT auth is preferred over sessions" > inbox/decision.txt
mkdir -p inbox/docs && cp architecture_notes.md inbox/docs/
# Agent auto-ingests recursively within 5-10 seconds
# Updates to existing files will semantically invalidate old memories and re-ingest automatically
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

## MCP Server

Always-On-Memory exposes a Model Context Protocol (MCP) server alongside its
REST API. Any MCP-compatible agent (Claude, GPT, Gemini, OpenClaw, etc.) can
use it to store and retrieve memories.

### Starting the MCP Server

The MCP server starts automatically with the agent. Configure it via `.env`
or command-line flags:

```bash
# Local (default): MCP on port 8765, REST on port 8888
python agent.py --watch ./inbox --port 8888 --mcp-port 8765

# Disable MCP server
python agent.py --mcp-port 0

# Custom host/port
python agent.py --mcp-host 127.0.0.1 --mcp-port 9000
```

### Authentication

Set `AOM_API_KEYS` to a comma-separated list of `name:key` pairs. Each key
is used by exactly one agent:

```bash
# In .env
AOM_API_KEYS=claude-desktop:your-secret-key,gpt-agent:another-key
```

All MCP requests must include `Authorization: Bearer <key>`. If `AOM_API_KEYS`
is unset, authentication is **disabled** (only safe for localhost deployments).

### Claude Desktop / Claude Code Configuration

Add to your Claude Desktop `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "always-on-memory": {
      "transport": {
        "type": "http",
        "url": "http://localhost:8765/mcp"
      },
      "headers": {
        "Authorization": "Bearer your-secret-key"
      }
    }
  }
}
```

### Docker

```bash
# Copy and edit the environment file
cp .env .env.local
# Set GOOGLE_API_KEY and AOM_API_KEYS in .env.local

docker compose --env-file .env.local up -d
```

Ports exposed: `8888` (REST API) and `8765` (MCP server).

### Available MCP Tools

| Tier | Tool | Description |
|------|------|-------------|
| Core | `remember(text, source?)` | Store information into persistent memory |
| Core | `recall(question)` | Search memories and synthesise an answer |
| Core | `status()` | Get memory system health statistics |
| Core | `forget(memory_id)` | Permanently delete a memory |
| Power | `list_memories(limit?)` | Browse memories ranked by importance |
| Power | `search_documents(query, limit?)` | Semantic search over indexed source files |
| Power | `consolidate()` | Trigger adversarial consolidation (lite models) |
| Power | `deep_reconsolidate()` | Full re-consolidation with smart models |
| Power | `export_memories(memory_ids?)` | Export MemCubes as portable JSON |
| Power | `import_memories(cubes)` | Import MemCube JSON |
| Power | `list_links()` | List code-memory structural links |
| Power | `reinforce(memory_id)` | Boost importance and reset decay clock |
| Power | `self_improve()` | Trigger self-improvement skill discovery |

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
| `/links` | GET | List structural links + grounding integrity (V3.3) |
| `/consolidate` | POST | Trigger manual consolidation |
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

When `WATCH_DIRS` is set, the agent periodically scans those directories for changes using `os.path.getmtime`. To avoid indexing while files are still being actively modified (e.g., by an LLM), it implements a **10-second debounce timer**.

- **Detection**: Checks for modifications every 5 seconds (`SCAN_INTERVAL`) using high-performance `os.scandir` walkers that prune ignored directories (e.g. `node_modules`) at the recursion point.
- **Background Threading**: FS-heavy operations are shifted to background threads to ensure the main Agent loop remains responsive to queries even during large indexing tasks.
- **Debounce**: Waits for a 10-second quiet window (`DEBOUNCE_INTERVAL`) after the last detected change before indexing.
- **Indexing**: Uses MD5 hashing to confirm changes before generating `sqlite-vec` embeddings (`gemini-embedding-2-preview`). 
- **Structural Chunking**: Code is partitioned using language-aware boundaries (functions/classes) rather than fixed character offsets, ensuring semantic units remain intact.
- **Lexical Symbol Index (LSI)**: Extract classes, functions, and variable signatures into a dedicated relational index for instant, exact-match code navigation.

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
| `MODEL` | `gemini-3.1-flash-lite-preview` | Lite model for ingest/consolidate/query |
| `SMART_MODEL` | `gemini-3-flash-preview` | Smart model for deep re-consolidation |
| `EMBEDDING_MODEL` | `gemini-embedding-2-preview` | Model for vector embeddings |
| `MEMORY_DB` | `memory.db` | SQLite database path |
| `RATE_LIMIT` | `15` | Max concurrent model requests |
| `WATCH_DIRS` | (empty) | Comma-separated dirs for Librarian mode |
| `IGNORE_DIRS` | (empty) | Extra directory names to skip |
| `SKILLS_DIR` | `.agents/skills` | Directory where skills are stored |
| `DEBOUNCE_INTERVAL` | `10` | Seconds to wait after change before indexing |
| `DRIFT_THRESHOLD` | `0.18` | Threshold for link evolution auditing |
| `PROMOTION_THRESHOLD` | `0.35` | Threshold for promoting code to semantic memory |
| `IDLE_THRESHOLD_MINUTES` | `30` | Idle time before AutoDream begins |
| `AOM_API_KEYS` | (empty) | `name:key` pairs for MCP authentication |

## Project Structure

The project has been refactored into focused, type-safe modules:

- `agent.py`: Principal entry point and background loop orchestrator. Powered by `asyncio.run()`.
- `agents_factory.py`: PydanticAI agent definitions and tool mapping.
- `config.py`: Centralized environment variable loading and settings.
- `database.py`: SQLite connection and `sqlite-vec` initialization.
- `librarian.py`: Semantic file search (Librarian) logic using high-performance `os.scandir` walkers.
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

## Research & Inspiration

This project integrates several cutting-edge concepts from the AI research community:

- **Always-On-Memory (MemOS)**: Inspired by the [MemTensor](https://github.com/MemTensor/MemOS) team's work on the first memory operating system for LLM agents (Stardust v2.0).
- **Adversarial Consolidation**: Based on [Anthropic's harness design](https://www.anthropic.com/engineering/harness-design-long-running-apps) for long-running application development.
- **AutoDream**: Derived from [Anthropic's](https://claudefa.st/blog/guide/mechanics/auto-dream) experimental background memory consolidation feature for Claude Code.
- **TurboQuant**: Inspired by [Google's TurboQuant](https://www.marktechpost.com/2026/03/25/google-introduces-turboquant-a-new-compression-algorithm-that-reduces-llm-key-value-cache-memory-by-6x-and-delivers-up-to-8x-speedup-all-with-zero-accuracy-loss/) — applies a seeded random orthogonal rotation before int8 scalar quantization, achieving ~75% storage reduction (float32 → int8) while preserving cosine similarity.

For detailed instructions on how to integrate your own agents with this memory layer, see the [Agent Memory Integration Guide](Agents.md).

## Credits

This project was derived from the [Google Cloud Platform Always-On-Memory Agent](https://github.com/GoogleCloudPlatform/generative-ai/tree/main/gemini/agents/always-on-memory-agent) and has been expanded with PydanticAI, Librarian mode (vector search), and rate-limiting resilience.

## License

MIT
