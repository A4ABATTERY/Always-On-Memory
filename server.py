"""
Server Module — Defines the HTTP API using aiohttp.
"""

from aiohttp import web
from typing import Any, Dict

from memory_store import (
    read_all_memories, get_memory_stats, delete_memory,
    # clear_all_memories is not yet moved
)
from librarian import search_documents

# Note: MemoryAgent and clear_all_memories will be passed or imported correctly.

def build_http(agent: Any, watch_path: str = "./inbox") -> web.Application:
    """Build the aiohttp application with all routes."""
    app = web.Application()

    async def handle_query(request: web.Request) -> web.Response:
        q = request.query.get("q", "").strip()
        if not q:
            return web.json_response({"error": "missing ?q= parameter"}, status=400)
        answer = await agent.query(q)
        return web.json_response({"question": q, "answer": answer})

    async def handle_ingest(request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        text = data.get("text", "").strip()
        if not text:
            return web.json_response({"error": "missing 'text' field"}, status=400)
        source = data.get("source", "api")
        result = await agent.ingest(text, source=source)
        return web.json_response({"status": "ingested", "response": result})

    async def handle_consolidate(request: web.Request) -> web.Response:
        result = await agent.consolidate()
        return web.json_response({"status": "done", "response": result})

    async def handle_reconsolidate(request: web.Request) -> web.Response:
        result = await agent.deep_reconsolidate()
        return web.json_response({"status": "done", "response": result})

    async def handle_improve(request: web.Request) -> web.Response:
        result = await agent.self_improve()
        return web.json_response({"status": "done", "response": result})

    async def handle_status(request: web.Request) -> web.Response:
        stats = get_memory_stats()
        return web.json_response(stats)

    async def handle_memories(request: web.Request) -> web.Response:
        data = read_all_memories()
        return web.json_response(data)

    async def handle_delete(request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        memory_id = data.get("memory_id")
        if not memory_id:
            return web.json_response({"error": "missing 'memory_id' field"}, status=400)
        result = delete_memory(int(memory_id))
        return web.json_response(result)

    async def handle_clear(request: web.Request) -> web.Response:
        # clear_all_memories will be handled by the agent or a separate function
        result = await agent.clear(watch_path=watch_path)
        return web.json_response(result)

    async def handle_search(request: web.Request) -> web.Response:
        q = request.query.get("q", "").strip()
        k = int(request.query.get("k", "5"))
        if not q:
            return web.json_response({"error": "missing ?q= parameter"}, status=400)
        results = await search_documents(q, k=k)
        return web.json_response(results)

    app.router.add_get("/query", handle_query)
    app.router.add_post("/ingest", handle_ingest)
    app.router.add_post("/consolidate", handle_consolidate)
    app.router.add_post("/reconsolidate", handle_reconsolidate)
    app.router.add_post("/improve", handle_improve)
    app.router.add_get("/status", handle_status)
    app.router.add_get("/memories", handle_memories)
    app.router.add_post("/delete", handle_delete)
    app.router.add_post("/clear", handle_clear)
    app.router.add_get("/search", handle_search)

    return app
