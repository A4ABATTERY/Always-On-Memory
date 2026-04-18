"""
mcp_server.py — FastMCP server for Always-On-Memory v3.

Exposes two tiers of MCP tools:
  Core (4):  remember, recall, status, forget
  Power (9): list_memories, search_documents, consolidate, deep_reconsolidate,
             export_memories, import_memories, list_links, reinforce, self_improve

All tools are LLM-agnostic: descriptions avoid Claude-specific language so they
work equally well with GPT, Gemini, and any other MCP-compatible agent.

Auth: every request must carry  Authorization: Bearer <key>  where <key> is
a value from AOM_API_KEYS env var (format: "name1:key1,name2:key2").
If AOM_API_KEYS is unset, auth is DISABLED (useful for localhost-only deployments,
risky for public/Docker deployments).
"""

import logging
from typing import Optional, List

from fastmcp import FastMCP

from mcp_auth import load_api_keys, APIKeyMiddleware

log = logging.getLogger("memory-agent.mcp")

# Global agent reference — set by run_mcp_server() before the server starts.
# Tools import this at call time (not at module load) to avoid circular imports.
_agent = None  # type: ignore[assignment]

# ─── FastMCP Instance ─────────────────────────────────────────────────────────

mcp = FastMCP(
    "Always-On-Memory",
    instructions=(
        "Always-On-Memory is a persistent, self-optimizing memory system. "
        "Use 'remember' to store information and 'recall' to retrieve it. "
        "Memories are automatically consolidated, clustered, and quality-scored "
        "in the background. Call 'status' to see the current state of the memory "
        "system. Use power tools for advanced operations like export, import, and "
        "manual consolidation triggers."
    ),
)

# ─── Core Semantic Tools ──────────────────────────────────────────────────────

@mcp.tool()
async def remember(text: str, source: str = "") -> str:
    """
    Store new information into persistent memory.

    Analyzes the input, extracts entities and topics, assigns an importance
    score, and writes a MemCube to the database. The memory is available for
    recall immediately.

    Args:
        text:   The information to remember. Can be facts, decisions, code
                patterns, observations, or any text content.
        source: Optional identifier for where this information came from
                (e.g. "code-review", "meeting-notes", "slack"). Defaults to "".

    Returns:
        Confirmation string with the assigned MemCube ID.
    """
    from mcp_auth import _caller_ctx
    origin = _caller_ctx.get()  # "mcp:{agent_name}" or "unknown" (no-auth mode)
    return await _agent.ingest(text, source=source, origin_platform=origin)


@mcp.tool()
async def recall(question: str) -> str:
    """
    Search stored memories and synthesize an answer to a question.

    Performs hybrid search (semantic vector search + keyword match) across
    all stored MemCubes and indexed source documents, then synthesizes a
    coherent answer with citations.

    Args:
        question: The question or topic to search for. Natural language queries
                  work best (e.g. "What authentication approach do we use?",
                  "Summarise recent decisions about the database schema").

    Returns:
        Synthesized answer with memory references.
    """
    return await _agent.query(question)


@mcp.tool()
async def status() -> dict:
    """
    Return memory system health statistics.

    Reports total memories, unconsolidated count, number of consolidations
    performed, indexed documents, and structural code-memory link counts.

    Returns:
        Dict with keys: total_memories (int), unconsolidated (int),
        consolidations (int), indexed_documents (int),
        structural_links (dict with active/historical_trace counts).
    """
    from memory_store import get_memory_stats
    return get_memory_stats()


@mcp.tool()
async def forget(memory_id: int) -> dict:
    """
    Permanently delete a specific memory by its numeric ID.

    This operation is irreversible. Use status() to get a count overview,
    or list_memories() to browse memories and find the ID to delete.

    Args:
        memory_id: The integer ID of the memory to delete (from list_memories()
                   or the 'id' field in status output).

    Returns:
        Dict with keys: status ("deleted" or "not_found"), memory_id (int).
    """
    from memory_store import delete_memory
    return delete_memory(memory_id)


# ─── Power Tools ─────────────────────────────────────────────────────────────

@mcp.tool()
async def list_memories(limit: int = 50) -> dict:
    """
    List stored memories ranked by composite score (importance × recency decay).

    Useful for browsing what the system knows, finding memory IDs for use with
    forget() or reinforce(), or auditing memory quality.

    Args:
        limit: Maximum number of memories to return (default 50, max 200).

    Returns:
        Dict with keys: memories (list of dicts with id, summary, sector,
        importance_score, composite_score, created_at, consolidated),
        count (int).
    """
    from memory_store import read_all_memories
    return read_all_memories(limit=limit)


@mcp.tool()
async def search_documents(query: str, limit: int = 5) -> dict:
    """
    Semantic search over indexed source code and documents.

    Searches the Librarian's vector index of watched directories. Returns
    the most semantically similar file chunks, ranked by cosine distance.
    Requires WATCH_DIRS to be configured and files to have been indexed.

    Args:
        query: Natural language description of the code or content to find.
        limit: Number of results to return (default 5).

    Returns:
        Dict with keys: results (list of dicts with path, snippet, distance,
        chunk_index), count (int).
    """
    from librarian import search_documents as _search
    return await _search(query, k=limit)


@mcp.tool()
async def consolidate() -> str:
    """
    Trigger adversarial consolidation of unconsolidated memories.

    Runs the Generator-Evaluator loop: the Generator synthesises a cluster of
    related memories into an Insight Cube; the Evaluator scores it for fidelity
    and completeness. Accepted syntheses are stored; source memories are marked
    consolidated. Uses lite models (fast, low-cost).

    Returns:
        Status string describing how many memories were processed.
    """
    return await _agent.consolidate()


@mcp.tool()
async def deep_reconsolidate() -> str:
    """
    Trigger full deep re-consolidation using smart (high-quality) models.

    Re-processes all memories including previously consolidated ones. Uses
    the most capable available models (higher cost, slower). Followed by
    a self-improvement audit. Intended for periodic quality maintenance.

    Returns:
        Status string describing the reconsolidation outcome.
    """
    return await _agent.deep_reconsolidate()


@mcp.tool()
async def export_memories(memory_ids: Optional[List[int]] = None) -> dict:
    """
    Export memories as portable MemCube JSON.

    Exported cubes can be imported into another AOM instance using
    import_memories(). Useful for backup, migration, or sharing memory
    state across systems. Includes all metadata but excludes embeddings
    (they will be re-generated on import).

    Args:
        memory_ids: Optional list of integer IDs to export. If omitted,
                    all memories are exported.

    Returns:
        Dict with keys: cubes (list of MemCube dicts), count (int).
    """
    from memory_store import export_cubes
    return export_cubes(memory_ids=memory_ids)


@mcp.tool()
async def import_memories(cubes: List[dict]) -> dict:
    """
    Import MemCube JSON into the memory system.

    Skips any cube whose cube_id already exists in the database (idempotent).
    Re-generates embeddings for each imported cube via the configured
    embedding model. Useful for restoring backups or merging two AOM instances.

    Args:
        cubes: List of MemCube dicts as produced by export_memories().
               Each dict must contain at least: cube_id, raw_text, summary.

    Returns:
        Dict with keys: status ("success"), imported (int count of new cubes).
    """
    from memory_store import import_cubes
    return await import_cubes(cubes)


@mcp.tool()
async def list_links() -> dict:
    """
    List all structural code-memory links and their integrity status.

    Structural links connect memories to the source files they were derived
    from. The Sync Auditor detects drift (code changes without memory update)
    and marks links as active or historical_trace.

    Returns:
        Dict with keys: links (list of dicts with memory_id, path,
        relationship, status, updated_at), count (int).
    """
    from memory_store import get_all_links
    return get_all_links()


@mcp.tool()
async def reinforce(memory_id: int) -> dict:
    """
    Increase the importance score of a memory and reset its decay clock.

    Memories decay in importance over time. Call reinforce() to confirm
    that a memory is still accurate and relevant — this boosts its score
    by 0.1 (capped at 1.0) and resets created_at so the decay restarts.

    Args:
        memory_id: The integer ID of the memory to reinforce.

    Returns:
        Dict with keys: status ("reinforced" or "not_found"),
        memory_id (int), new_importance (float).
    """
    from memory_store import reinforce_memory
    return reinforce_memory(memory_id)


@mcp.tool()
async def self_improve() -> str:
    """
    Trigger the self-improvement agent to discover patterns and write skill files.

    Analyses reflection and episodic memories to identify recurring patterns,
    successful strategies, and domain knowledge. Writes discovered skills to
    the configured SKILLS_DIR as structured SKILL.md files.

    Returns:
        Status string describing how many skills were written or updated.
    """
    return await _agent.self_improve()


# ─── Server Startup ───────────────────────────────────────────────────────────

async def run_mcp_server(
    agent_instance,
    host: str = "0.0.0.0",
    port: int = 8765,
) -> None:
    """
    Start the FastMCP HTTP server as an asyncio-compatible coroutine.

    Intended to be launched as an asyncio task alongside the REST server:
        asyncio.create_task(run_mcp_server(agent, host, port))

    Auth: if AOM_API_KEYS is set, injects APIKeyMiddleware via http_app(middleware=[...])
    so the Starlette lifespan (and FastMCP's StreamableHTTPSessionManager) is preserved.
    Wrapping the returned app externally with BaseHTTPMiddleware discards the lifespan
    and breaks all MCP requests — always use the middleware= parameter instead.

    lifespan="on" is mandatory in uvicorn.Config: FastMCP's StreamableHTTPSessionManager
    is initialised inside the Starlette lifespan; omitting it causes every request to fail
    with "StreamableHTTPSessionManager task group was not initialized."

    Args:
        agent_instance: The MemoryAgent instance shared with the REST server.
        host: Bind address (default "0.0.0.0").
        port: TCP port for the MCP server (default 8765).
    """
    import asyncio as _asyncio
    import os as _os
    import uvicorn
    from starlette.middleware import Middleware

    global _agent
    _agent = agent_instance

    api_keys = load_api_keys()
    no_auth_explicitly_allowed = _os.getenv("AOM_MCP_NO_AUTH", "").lower() in ("1", "true", "yes")

    if not api_keys:
        if not no_auth_explicitly_allowed:
            raise RuntimeError(
                "MCP server requires authentication. Set AOM_API_KEYS=name:key,... "
                "or set AOM_MCP_NO_AUTH=true to explicitly disable auth (localhost only)."
            )
        log.warning(
            "⚠️  MCP server starting WITHOUT authentication (AOM_MCP_NO_AUTH=true). "
            "Only safe for localhost-only deployments."
        )

    # Pass auth middleware via http_app(middleware=[...]) so FastMCP wires it
    # inside the Starlette app before the lifespan is attached.
    # DO NOT wrap the returned app externally — that discards the lifespan.
    middleware = [Middleware(APIKeyMiddleware, api_keys=api_keys)] if api_keys else []

    asgi_app = mcp.http_app(transport="streamable-http", middleware=middleware)

    mcp_uvicorn_config = uvicorn.Config(
        asgi_app,
        host=host,
        port=port,
        log_level="info",
        lifespan="on",     # REQUIRED: initialises StreamableHTTPSessionManager
        access_log=False,  # AOM uses its own logging
        # install_handlers removed in uvicorn 0.44.0; signal handling managed via
        # the _watch_shutdown task below that propagates agent shutdown to uvicorn.
    )
    server = uvicorn.Server(mcp_uvicorn_config)

    log.info(f"🔌 MCP server starting on {host}:{port} (streamable-http transport)")

    # Bidirectional shutdown wiring: when the agent's _shutdown_event fires
    # (e.g. Ctrl+C caught by agent.py's loop.add_signal_handler), propagate
    # to uvicorn so both servers stop together.
    # Background: uvicorn.Server.serve() calls capture_signals() which installs
    # its own SIGINT/SIGTERM handlers via signal.signal(), overwriting the asyncio
    # handlers registered in agent.py lines 673-674. The watcher below ensures
    # the agent-side shutdown always reaches uvicorn regardless of which side
    # catches the signal first.
    from config import get_shutdown_event

    async def _watch_shutdown() -> None:
        await get_shutdown_event().wait()
        server.should_exit = True

    _watch_task = _asyncio.create_task(_watch_shutdown())
    try:
        await server.serve()
    finally:
        _watch_task.cancel()
