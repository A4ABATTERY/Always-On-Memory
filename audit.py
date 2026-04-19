"""
Audit Module — Immutable event log for all memory mutations.

Every create, repair, and delete operation writes:
  1. A row to the `memory_audit_log` SQLite table (for structured queries via /audit).
  2. A JSON line to `AOM_AUDIT_LOG_DIR/YYYY-MM-DD.jsonl` (for external consumption).

Both writes are fire-and-forget: errors are logged but never raised to callers.
JSONL writing is offloaded with asyncio.to_thread() to avoid blocking the event loop.
"""

import asyncio
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

log = logging.getLogger("memory-agent.audit")


async def write_audit_event(
    cube_id: str,
    event_type: str,
    actor: str,
    before_text: Optional[str] = None,
    after_text: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist an audit event to both the SQLite table and a daily JSONL file.

    Args:
        cube_id:     The MemCube UUID this event belongs to.
        event_type:  One of: 'created', 'repaired', 'deleted', 'consolidated'.
        actor:       origin_platform value identifying the agent/process that made the change.
        before_text: raw_text BEFORE the change (None for 'created').
        after_text:  raw_text AFTER the change (None for 'deleted').
        metadata:    Optional JSON-serialisable dict (e.g. consolidation_id, score).
    """
    now = datetime.now(timezone.utc).isoformat()
    event: Dict[str, Any] = {
        "cube_id": cube_id,
        "event_type": event_type,
        "actor": actor,
        "before_text": before_text,
        "after_text": after_text,
        "metadata": metadata or {},
        "created_at": now,
    }

    # --- DB write (synchronous, offloaded) ---
    async def _db_write() -> None:
        try:
            from database import db_session
            with db_session() as db:
                db.execute(
                    """INSERT INTO memory_audit_log
                       (cube_id, event_type, actor, before_text, after_text, metadata, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        cube_id, event_type, actor,
                        before_text, after_text,
                        json.dumps(metadata or {}),
                        now,
                    ),
                )
                db.commit()
        except Exception as exc:
            log.error(f"audit: DB write failed for {event_type}@{cube_id}: {exc}")

    # --- JSONL write (synchronous, offloaded) ---
    async def _jsonl_write() -> None:
        from config import AOM_AUDIT_LOG_DIR
        if not AOM_AUDIT_LOG_DIR:
            return
        try:
            log_dir = Path(AOM_AUDIT_LOG_DIR)
            log_path = log_dir / f"{date.today().isoformat()}.jsonl"

            def _write() -> None:
                log_dir.mkdir(parents=True, exist_ok=True)
                with log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")

            await asyncio.to_thread(_write)
        except Exception as exc:
            log.error(f"audit: JSONL write failed for {event_type}@{cube_id}: {exc}")

    await asyncio.gather(_db_write(), _jsonl_write())


async def write_conversation_log(
    agent: str,
    model: str,
    source: str,
    origin_platform: str,
    input_tokens: int,
    output_tokens: int,
    memories_created: Optional[list] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one line to the daily JSONL log capturing an agent's conversation metadata.

    This is separate from write_audit_event() — it records *agent calls*
    (model, token cost, what was produced) rather than memory state transitions.

    Args:
        agent:            Agent name (e.g. 'ingest', 'generator', 'evaluator').
        model:            Model string used for this call.
        source:           Source file or context identifier.
        origin_platform:  Caller identity (e.g. 'inbox-watcher', 'mcp:claude-code').
        input_tokens:     Tokens consumed for the prompt.
        output_tokens:    Tokens generated in the response.
        memories_created: List of cube_id strings created by this call.
        metadata:         Any extra context (consolidation IDs, quality scores, etc.).
    """
    from config import AOM_AUDIT_LOG_DIR
    if not AOM_AUDIT_LOG_DIR:
        return

    now = datetime.now(timezone.utc).isoformat()
    entry: Dict[str, Any] = {
        "ts": now,
        "agent": agent,
        "model": model,
        "source": source,
        "origin_platform": origin_platform,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "memories_created": memories_created or [],
        **(metadata or {}),
    }

    try:
        log_dir = Path(AOM_AUDIT_LOG_DIR)
        log_path = log_dir / f"{date.today().isoformat()}.jsonl"

        def _write() -> None:
            log_dir.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        await asyncio.to_thread(_write)
    except Exception as exc:
        log.error(f"audit: conversation log write failed for agent={agent}: {exc}")
