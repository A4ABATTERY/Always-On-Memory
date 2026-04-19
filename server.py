"""
Server Module — Defines the HTTP API using aiohttp.
"""

from aiohttp import web
from typing import Any

from memory_store import (
    read_all_memories, get_memory_stats, delete_memory,
    get_all_links
)

from librarian import search_documents, verify_watch_dirs
from config import INBOX_DIR

# Note: MemoryAgent and clear_all_memories will be passed or imported correctly.

def build_http(agent: Any, watch_path: str = INBOX_DIR) -> web.Application:
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

    async def handle_export_cubes(request: web.Request) -> web.Response:
        """Export all memories as portable MemCube JSON."""
        from memory_store import export_cubes
        ids_param = request.query.get("ids", "")
        ids = [int(i) for i in ids_param.split(",") if i.strip()] if ids_param else None
        result = export_cubes(memory_ids=ids)
        return web.json_response(result)

    async def handle_import_cubes(request: web.Request) -> web.Response:
        """Import MemCube JSON into the database."""
        from memory_store import import_cubes
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        
        # Handle both {"cubes": [...]} and [...]
        if isinstance(data, list):
            cubes = data
        elif isinstance(data, dict):
            cubes = data.get("cubes", [])
        else:
            return web.json_response({"error": "invalid format: expected list or object with 'cubes' array"}, status=400)

        if not cubes:
            return web.json_response({"error": "missing or empty 'cubes' array"}, status=400)
            
        result = await import_cubes(cubes)
        return web.json_response(result)

    async def handle_links(request: web.Request) -> web.Response:
        """List all code-memory structural links."""
        result = get_all_links()
        return web.json_response(result)

    async def handle_verify(request: web.Request) -> web.Response:
        """Manually trigger a verification scan of WATCH_DIRS.

        Detects files that are missing or incompletely indexed and retries them
        up to 3 times per agent lifetime. Shares retry state with the background
        verification_loop so counts are consistent.
        """
        result = await verify_watch_dirs(
            agent.verify_retry_state,
            on_drift_detected=agent.push_sync_task,
            on_promotion_triggered=agent.promote_file,
        )
        return web.json_response({
            "status": "ok",
            "missing": len(result["missing"]),
            "incomplete": len(result["incomplete"]),
            "retried_ok": len(result["retried_ok"]),
            "retried_failed": len(result["retried_failed"]),
            "permanent_failures": len(result["permanent_failures"]),
            "details": result,
        })

    async def handle_audit(request: web.Request) -> web.Response:
        """Return audit history for a MemCube or all events since a given date.

        Query params (at least one required):
          ?cube_id=<uuid>      — full history for one MemCube
          ?since=<ISO-date>    — all events on or after the given timestamp/date
        """
        from database import db_session
        import json as _json

        cube_id = request.query.get("cube_id", "").strip()
        since = request.query.get("since", "").strip()

        if not cube_id and not since:
            return web.json_response(
                {"error": "provide ?cube_id=<id> or ?since=<ISO-date>"},
                status=400,
            )

        with db_session() as db:
            if cube_id and since:
                rows = db.execute(
                    "SELECT * FROM memory_audit_log WHERE cube_id = ? AND created_at >= ? ORDER BY created_at ASC",
                    (cube_id, since),
                ).fetchall()
            elif cube_id:
                rows = db.execute(
                    "SELECT * FROM memory_audit_log WHERE cube_id = ? ORDER BY created_at ASC",
                    (cube_id,),
                ).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM memory_audit_log WHERE created_at >= ? ORDER BY created_at ASC",
                    (since,),
                ).fetchall()

        events = [
            {
                "id": r["id"],
                "cube_id": r["cube_id"],
                "event_type": r["event_type"],
                "actor": r["actor"],
                "before_text": r["before_text"],
                "after_text": r["after_text"],
                "metadata": _json.loads(r["metadata"] or "{}"),
                "created_at": r["created_at"],
            }
            for r in rows
        ]
        return web.json_response({"events": events, "count": len(events)})

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
    app.router.add_get("/export_cubes", handle_export_cubes)
    app.router.add_post("/import_cubes", handle_import_cubes)
    app.router.add_get("/links", handle_links)
    app.router.add_post("/verify", handle_verify)
    app.router.add_get("/audit", handle_audit)

    return app
