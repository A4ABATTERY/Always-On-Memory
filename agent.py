"""
Agent Memory Layer — Always-On PydanticAI Agent Entry Point.

Refactored into a modular architecture for better maintainability and type safety.
"""

import argparse
import asyncio
import logging
import shutil
import signal
import time
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, cast

from pydantic_ai import Agent
from aiohttp import web
from config import (
    SMART_MODEL,
    TEXT_EXTENSIONS, ALL_SUPPORTED, MEDIA_EXTENSIONS,
    get_shutdown_event, IDLE_THRESHOLD_MINUTES, AUTODREAM_CHECK_INTERVAL,
    HAS_SQLITE_VEC, INBOX_DIR
)
from models import SynthesisResult, AuditResult, EvalResult

# Module-level reference — set to the real event at the start of main_async()
_shutdown_event: Optional[asyncio.Event] = None
from database import init_db, db_session
from agents_factory import build_agents
from utils import retry_with_backoff
from memory_store import (
    read_all_memories
)
from librarian import librarian_loop
from server import build_http

try:
    import google.genai as genai
    HAS_GENAI = True
except ImportError:
    genai = cast(Any, None)
    HAS_GENAI = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="[%H:%M]",
)
log = logging.getLogger("memory-agent")

# ─── MemoryAgent Orchestrator ──────────────────────────────────

# Module-level activity tracker
_last_activity_time: float = time.time()

def record_activity():
    """Call this from any endpoint or ingestion to track system activity."""
    global _last_activity_time
    _last_activity_time = time.time()

class MemoryAgent:
    """Orchestrates the various PydanticAI agents and memory operations."""
    
    def __init__(self) -> None:
        (
            self.ingest_agent,
            self.generator_lite,
            self.evaluator_lite,
            self.generator_smart,
            self.evaluator_smart,
            self.query_agent,
            self.self_improvement_agent,
            self.sync_agent,
        ) = build_agents()
        
        self.ingest_agent: Agent[None, str] = self.ingest_agent
        self.generator_lite: Agent[None, SynthesisResult] = self.generator_lite
        self.evaluator_lite: Agent[None, EvalResult] = self.evaluator_lite
        self.generator_smart: Agent[None, SynthesisResult] = self.generator_smart
        self.evaluator_smart: Agent[None, EvalResult] = self.evaluator_smart
        self.query_agent: Agent[None, str] = self.query_agent
        self.self_improvement_agent: Agent[None, str] = self.self_improvement_agent
        self.sync_agent: Agent[None, AuditResult] = self.sync_agent
        
        self.client = genai.Client() if HAS_GENAI else None
        self.sync_queue: asyncio.Queue = asyncio.Queue()

    async def push_sync_task(self, path: str, memory_id: int):
        """Callback for Librarian to push drift detection tasks."""
        await self.sync_queue.put({"path": path, "memory_id": memory_id})
        log.debug(f"📤 Pushed sync task for memory #{memory_id} on '{path}'")

    async def _audit_link(self, path: str, memory_id: int):
        """Use the Sync Agent to audit a link after drift is detected."""
        from librarian import read_document
        from memory_store import update_link_status
        from models import AuditResult

        # 1. Get current code snippet
        doc = read_document(path)
        if "error" in doc:
            log.warning(f"Sync audit failed: {doc['error']}")
            raise Exception(f"Document read error: {doc['error']}")

        # 2. Get memory insight
        with db_session() as db:
            row = db.execute("SELECT summary, raw_text FROM memories WHERE id = ?", (memory_id,)).fetchone()
        
        if not row:
            return
            
        insight = f"Summary: {row['summary']}\nDetails: {row['raw_text']}"
        code_snippet = doc["content"][:4000] # Limit for Lite agent

        # 3. Ask Sync Agent (Structured Result)
        log.info(f"⚖️ Auditing memory #{memory_id} against '{path}'...")
        result = await retry_with_backoff(
            self.sync_agent.run,
            f"MEMORY INSIGHT:\n{insight}\n\nCODE SNIPPET:\n{code_snippet}",
            shutdown_event=_shutdown_event
        )
        
        data: AuditResult = result.output
        status = data.status.upper()
        reason = data.reason
        suggested = data.suggested_update
        
        if status == "HISTORICAL":
            log.info(f"📜 Link evolved to HISTORICAL for memory #{memory_id}: {reason}")
            update_link_status(memory_id, path, "historical_trace")
        elif status == "REPAIR" and suggested:
            log.info(f"🛠️  Repairing memory #{memory_id}: {reason}")
            from memory_store import repair_memory
            repair_memory(memory_id, suggested)
            update_link_status(memory_id, path, "active")
        else:
            log.info(f"✅ Link remains ACTIVE for memory #{memory_id}: {reason}")
            update_link_status(memory_id, path, "active")

    async def sync_worker_loop(self):
        """Worker that consumes the sync queue with transient error retries."""
        log.info("⛓️ Sync worker loop started.")
        retry_counts = {} # path:memory_id -> count
        MAX_RETRIES = 3
        
        while _shutdown_event and not _shutdown_event.is_set():
            try:
                task = await asyncio.wait_for(self.sync_queue.get(), timeout=5.0)
                task_key = f"{task['path']}:{task['memory_id']}"
                
                try:
                    await self._audit_link(task["path"], task["memory_id"])
                    if task_key in retry_counts:
                        del retry_counts[task_key]
                except Exception as e:
                    count = retry_counts.get(task_key, 0) + 1
                    if count <= MAX_RETRIES:
                        log.warning(f"⚠️ Sync audit failed for {task_key} (attempt {count}/{MAX_RETRIES}): {e}. Re-queueing...")
                        retry_counts[task_key] = count
                        await self.sync_queue.put(task)
                    else:
                        log.error(f"❌ Sync audit permanently failed for {task_key} after {MAX_RETRIES} retries: {e}")
                        if task_key in retry_counts:
                            del retry_counts[task_key]
                
                self.sync_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                log.error(f"Sync worker loop error: {e}")

    async def ingest_file(self, file_path: Path) -> str:
# ... (rest of the file)

        """Ingest a text-based file from the inbox."""
        record_activity()
        suffix = file_path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")[:10000]
                if text.strip():
                    return await self.ingest(text, source=file_path.name)
            except Exception as e:
                log.error(f"Failed to read {file_path}: {e}")
        return f"Skipped: unsupported or empty file {file_path.name}"

    async def ingest(self, text: str, source: str = "") -> str:
        record_activity()
        log.info(f"📥 Analyzing {'file: ' + source if source else 'text content'} ({len(text)} chars)...")
        msg = f"Remember this information (source: {source}):\n\n{text}" if source else f"Remember this information:\n\n{text}"

        # Record timestamp before agent call so we can verify persistence afterward.
        from datetime import datetime, timezone
        before_ts = datetime.now(timezone.utc).isoformat()

        result = await retry_with_backoff(self.ingest_agent.run, msg, shutdown_event=_shutdown_event)

        usage = result.usage()
        log.info(f"📥 Ingested: {usage.total_tokens} tokens | {usage.input_tokens} req | {usage.output_tokens} res")

        # Reflexive-loop guard: verify the agent actually called store_memory.
        # If no new memory was written, fall back to direct persistence.
        from memory_store import store_memory as _store_memory
        with db_session() as db:
            new_row = db.execute(
                "SELECT id FROM memories WHERE created_at >= ? LIMIT 1", (before_ts,)
            ).fetchone()

        if not new_row:
            log.warning(
                "⚠️ Ingest agent did not call store_memory — falling back to direct persistence."
            )
            fallback = await _store_memory(
                raw_text=text,
                summary=f"Auto-ingested from {source}" if source else "Auto-ingested content",
                entities=[],
                topics=[],
                importance_score=0.5,
                sector="episodic",
                source=source or "ingest-fallback",
            )
            return f"Stored via fallback as MemCube #{fallback['memory_id']}"

        return result.output

    async def adversarial_consolidation(
        self,
        generator: Agent[None, SynthesisResult],
        evaluator: Agent[None, EvalResult],
        raw_memories_text: str,
        max_attempts: int = 3,
        quality_threshold: float = 0.85,
    ) -> dict:
        """
        Generator-Evaluator adversarial loop using PydanticAI structured results.
        """
        feedback = "No feedback yet. Generate the initial synthesis."
        for attempt in range(max_attempts):
            # Step 1: Generator creates synthesis (Structured Result)
            gen_result = await retry_with_backoff(
                generator.run,
                f"Synthesize these memories. Previous feedback: {feedback}\n\nMemories:\n{raw_memories_text}",
                shutdown_event=_shutdown_event,
            )
            draft_data: SynthesisResult = gen_result.output
            
            # Step 2: Evaluator grades it (Structured Result)
            eval_result = await retry_with_backoff(
                evaluator.run,
                f"Source memories:\n{raw_memories_text}\n\nDraft synthesis:\n{draft_data.model_dump_json()}\n\n"
                "Grade strictly. Output JSON ONLY with score, fidelity, completeness, redundancy_removed, feedback.",
                shutdown_event=_shutdown_event,
            )
            
            eval_data: EvalResult = eval_result.output
            score = eval_data.score
            feedback = eval_data.feedback
            
            log.info(f"🔄 Consolidation attempt {attempt+1}: score={score:.2f}")
            
            if score >= quality_threshold:
                log.info(f"✅ Consolidation approved (score={score:.2f})")
                return draft_data.model_dump()
            
            log.info(f"⚠️ Below threshold ({quality_threshold}). Feedback: {feedback[:100]}...")
        
        log.warning(f"❌ Consolidation failed after {max_attempts} attempts.")
        raise Exception("Consolidation quality threshold not met after max attempts.")

    async def consolidate(self) -> str:
        log.info("🔄 Running periodic adversarial consolidation...")
        from memory_store import read_unconsolidated_memories, store_consolidation, store_memory
        data = read_unconsolidated_memories()
        if data["count"] < 2:
            return "Nothing to consolidate"
        
        import json
        memories_text = json.dumps(data["memories"], indent=2)
        
        try:
            # 1. Run adversarial loop
            result_data = await self.adversarial_consolidation(
                self.generator_lite, self.evaluator_lite, memories_text
            )
            
            # 2. Persist consolidation record (marks old memories as consolidated)
            store_consolidation(
                source_ids=result_data["source_ids"],
                summary=result_data["summary"],
                insight=result_data["insight"],
                connections=result_data.get("connections", [])
            )
            
            # 3. Create new Insight MemCube
            new_cube = await store_memory(
                raw_text=result_data["insight"],
                summary=result_data["summary"],
                entities=[], # Could be extracted if needed
                topics=["consolidated-insight"],
                connections=result_data.get("connections", []),
                importance_score=0.8,
                sector="semantic",
                source="adversarial-consolidation"
            )
            
            return f"Consolidated {len(result_data['source_ids'])} memories into Insight Cube #{new_cube['memory_id']}"
        except Exception as e:
            log.error(f"Adversarial consolidation failed: {e}")
            return f"Consolidation failed: {e}"

    async def query(self, question: str) -> str:
        record_activity()
        log.info(f"🔍 Processing query: '{question}'")
        result = await retry_with_backoff(
            self.query_agent.run,
            f"Based on my memories, answer: {question}",
            shutdown_event=_shutdown_event
        )
        
        output = result.output
        memory_refs = output.count("[Memory")
        file_refs = output.count("/home/") + output.count("./") + output.count("Relevant Files")
        
        usage = result.usage()
        log.info(f"🔍 Answered: {usage.total_tokens} tokens | {memory_refs} refs | {file_refs} files")
        return output

    async def deep_reconsolidate(self) -> str:
        log.info(f"🧠 Running deep adversarial re-consolidation using {SMART_MODEL}...")
        from memory_store import store_consolidation, store_memory
        data = read_all_memories()
        
        import json
        memories_text = json.dumps(data["memories"], indent=2)
        
        try:
            # 1. Run adversarial loop with smart models
            result_data = await self.adversarial_consolidation(
                self.generator_smart, self.evaluator_smart, memories_text,
                max_attempts=3, quality_threshold=0.85
            )
            
            # 2. Persist consolidation
            store_consolidation(
                source_ids=result_data["source_ids"],
                summary=result_data["summary"],
                insight=result_data["insight"],
                connections=result_data.get("connections", [])
            )
            
            # 3. Create high-fidelity Insight MemCube
            new_cube = await store_memory(
                raw_text=result_data["insight"],
                summary=result_data["summary"],
                entities=[],
                topics=["deep-insight", "architectural-consensus"],
                connections=result_data.get("connections", []),
                importance_score=0.9,
                sector="semantic",
                source="deep-reconsolidation"
            )
            
            # After consolidation, run self-improvement
            await self.self_improve()
            return f"Deep consolidation complete. Created Insight Cube #{new_cube['memory_id']}"
        except Exception as e:
            log.error(f"Deep adversarial consolidation failed: {e}")
            return f"Deep consolidation failed: {e}"

    async def self_improve(self) -> str:
        log.info(f"🧬 Running self-improvement audit using {SMART_MODEL}...")
        result = await retry_with_backoff(
            self.self_improvement_agent.run,
            "Audit recent reflection and episodic memories to discover or refine skills. ",
            shutdown_event=_shutdown_event
        )
        usage = result.usage()
        log.info(f"🧬 Self-improvement audit complete: {usage.total_tokens} tokens used")
        return result.output

    async def clear(self, watch_path: str) -> dict:
        """Full reset of memories and inbox."""
        with db_session() as db:
            mem_count = db.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
            db.execute("DELETE FROM memories")
            db.execute("DELETE FROM consolidations")
            db.execute("DELETE FROM processed_files")
            db.execute("DELETE FROM documents") # Clear Librarian docs
            if HAS_SQLITE_VEC:
                try:
                    db.execute("DELETE FROM vec_documents") # Clear Librarian vectors
                except Exception:
                    pass
            db.commit()

        files_deleted = 0
        folder = Path(watch_path)
        if folder.is_dir():
            for f in folder.iterdir():
                if f.name.startswith("."):
                    continue
                try:
                    if f.is_file():
                        f.unlink()
                        files_deleted += 1
                    elif f.is_dir():
                        shutil.rmtree(f)
                        files_deleted += 1
                except OSError as e:
                    log.error(f"Failed to delete {f.name}: {e}")

        log.info(f"🗑️  Cleared all {mem_count} memories, deleted {files_deleted} inbox files")
        record_activity()
        return {"status": "cleared", "memories_deleted": mem_count, "files_deleted": files_deleted}

# ─── Background Loops ──────────────────────────────────────────

def system_is_idle(threshold_minutes: int = 30) -> bool:
    """Check if the system has been idle for threshold_minutes."""
    return (time.time() - _last_activity_time) > (threshold_minutes * 60)

async def autodream_loop(agent: MemoryAgent, check_interval: int = 300):
    """
    AutoDream: active idle-time memory optimization.
    Activates only when system has been idle for IDLE_THRESHOLD_MINUTES.
    Performs: importance decay → redundancy pruning → topic clustering → adversarial consolidation.
    """
    
    log.info(f"💤 AutoDream: checking every {check_interval}s, triggers after {IDLE_THRESHOLD_MINUTES}min idle")
    
    while _shutdown_event and not _shutdown_event.is_set():
        try:
            await asyncio.wait_for(_shutdown_event.wait(), timeout=check_interval)
            break
        except asyncio.TimeoutError:
            pass
        
        if not system_is_idle(IDLE_THRESHOLD_MINUTES):
            continue
        
        log.info("💤 Entering AutoDream sequence...")
        
        try:
            # Step 1: Importance decay (consolidated into dream)
            await _dream_decay()
            
            # Step 2: Redundancy pruning & Topic-based clustering
            await _dream_reorganize(agent)
            
            log.info("💤 AutoDream complete. Memory state optimized.")
        except Exception as e:
            log.error(f"AutoDream error: {e}")

async def _dream_decay():
    """Prune memories that have decayed below threshold."""
    with db_session() as db:
        rows = db.execute("SELECT id, importance_score, created_at FROM memories WHERE consolidated = 0").fetchall()
        for r in rows:
            created_dt = datetime.fromisoformat(r["created_at"])
            if created_dt.tzinfo is None:
                created_dt = created_dt.replace(tzinfo=timezone.utc)
            
            age_hours = (datetime.now(timezone.utc).timestamp() - created_dt.timestamp()) / 3600.0
            if age_hours > 24.0:
                new_importance = r["importance_score"] - 0.05
                if new_importance < 0.1:
                    log.info(f"💤 Dream: deleting decayed memory #{r['id']}")
                    db.execute("DELETE FROM memories WHERE id = ?", (r["id"],))
                    db.execute("DELETE FROM vec_memories WHERE memory_id = ?", (r["id"],))
                else:
                    db.execute("UPDATE memories SET importance_score = ? WHERE id = ?", (new_importance, r["id"]))
        db.commit()

async def _dream_reorganize(agent: MemoryAgent):
    """Cluster related memories using embeddings and trigger adversarial consolidation for large clusters."""
    from memory_store import (
        read_unconsolidated_with_embeddings, cluster_memories_by_embedding,
        store_consolidation, store_memory,
    )

    # Limit matches standard consolidate() to avoid over-aggressive idle reorganization.
    memories = read_unconsolidated_with_embeddings(limit=30)
    if len(memories) < 3:
        log.debug(f"💤 Dream: Not enough memories for clustering ({len(memories)}/3)")
        return

    # Use embedding-based clustering
    clusters = cluster_memories_by_embedding(memories, threshold=0.75)

    import json
    for cluster_name, cluster_members in clusters.items():
        if len(cluster_members) >= 3:
            log.info(f"💤 Dream: compressing {cluster_name} ({len(cluster_members)} memories)")
            cluster_text = json.dumps(cluster_members, indent=2)
            try:
                result_data = await agent.adversarial_consolidation(
                    agent.generator_smart,
                    agent.evaluator_smart,
                    cluster_text,
                    quality_threshold=0.9,
                )
                # Persist: mark sources consolidated and write the Insight Cube.
                store_consolidation(
                    source_ids=result_data["source_ids"],
                    summary=result_data["summary"],
                    insight=result_data["insight"],
                    connections=result_data.get("connections", []),
                )
                new_cube = await store_memory(
                    raw_text=result_data["insight"],
                    summary=result_data["summary"],
                    entities=[],
                    topics=["dream-insight"],
                    connections=result_data.get("connections", []),
                    importance_score=0.85,
                    sector="semantic",
                    source="autodream-consolidation",
                )
                log.info(f"💤 Dream: created Insight Cube #{new_cube['memory_id']} from {cluster_name}")
            except Exception as e:
                log.debug(f"Dream cluster consolidation failed: {e}")

async def watch_folder(agent: MemoryAgent, folder: Path, poll_interval: int = 5):
    """Watch a folder for new files and ingest them, tracking hash updates."""
    folder.mkdir(parents=True, exist_ok=True)
    log.info(f"👁️  Watching: {folder}/")

    while _shutdown_event and not _shutdown_event.is_set():
        try:
            with db_session() as db:
                rows = db.execute("SELECT path, content_hash FROM processed_files").fetchall()
                hash_map = {r["path"]: r["content_hash"] for r in rows}

            files = []
            for f in folder.rglob('*'):
                if f.is_file() and not f.name.startswith("."):
                    files.append(f)
            files.sort()

            for f in files:
                if _shutdown_event.is_set():
                    break
                
                suffix = f.suffix.lower()
                if suffix not in ALL_SUPPORTED:
                    continue
                
                rel_path = str(f.relative_to(folder)).replace('\\', '/')
                
                content_bytes = None
                try:
                    content_bytes = f.read_bytes()
                except Exception as e:
                    log.error(f"Cannot read file {f}: {e}")
                    continue

                if not content_bytes:
                    continue

                new_hash = hashlib.md5(content_bytes).hexdigest()
                last_hash = hash_map.get(rel_path)
                
                if last_hash == new_hash:
                    continue

                is_update = last_hash is not None

                try:
                    rolled_back_mems = []
                    from memory_store import get_memories_by_source, update_link_status, update_memory_validity
                    now_iso = datetime.now(timezone.utc).isoformat()
                    
                    if is_update:
                        log.info(f"🔄 File update detected for {rel_path}. Invalidating superseded memories.")
                        old_memory_ids = get_memories_by_source(rel_path)
                        for mid in old_memory_ids:
                            update_memory_validity(mid, now_iso)
                            update_link_status(mid, rel_path, "historical_trace")
                            rolled_back_mems.append(mid)

                    before_ts = datetime.now(timezone.utc).isoformat()

                    success = False
                    if suffix in TEXT_EXTENSIONS:
                        log.info(f"📄 Processing text file: {rel_path} (update: {is_update})")
                        text = content_bytes.decode('utf-8', errors='replace')[:10000]
                        if text.strip():
                            await agent.ingest(text, source=rel_path)
                            success = True
                    elif suffix in MEDIA_EXTENSIONS:
                        log.info(f"🖼️  Processing media file: {rel_path} (update: {is_update})")
                        await agent.ingest(f"New media file found: {f.name}", source=rel_path)
                        success = True

                    if success:
                        with db_session() as db:
                            if is_update:
                                db.execute(
                                    "UPDATE processed_files SET content_hash = ?, prev_hash = ?, processed_at = ? WHERE path = ?",
                                    (new_hash, last_hash, now_iso, rel_path)
                                )
                            else:
                                db.execute(
                                    "INSERT INTO processed_files (path, processed_at, content_hash) VALUES (?, ?, ?)",
                                    (rel_path, now_iso, new_hash)
                                )
                            db.commit()
                        hash_map[rel_path] = new_hash

                except Exception as file_err:
                    log.error(f"Error ingesting {f.name}: {file_err}")
                    if is_update and rolled_back_mems:
                        log.warning(f"Rolling back valid_to stamps for {rel_path}")
                        for mid in rolled_back_mems:
                            update_memory_validity(mid, None)
                            update_link_status(mid, rel_path, "active")
                            
                        with db_session() as db:
                            new_mems = db.execute("SELECT id FROM memories WHERE source = ? AND created_at >= ?", (rel_path, before_ts)).fetchall()
                            for nm in new_mems:
                                db.execute("DELETE FROM memories WHERE id = ?", (nm["id"],))
                                db.execute("DELETE FROM vec_memories WHERE memory_id = ?", (nm["id"],))
                            db.commit()

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Watch error: {e}")

        try:
            await asyncio.wait_for(_shutdown_event.wait(), timeout=poll_interval)
            break
        except asyncio.TimeoutError:
            pass

async def consolidation_loop(agent: MemoryAgent, interval_minutes: int = 30):
    """Run consolidation periodically."""
    log.info(f"🔄 Consolidation: every {interval_minutes} minutes")
    while _shutdown_event and not _shutdown_event.is_set():
        try:
            await asyncio.wait_for(_shutdown_event.wait(), timeout=interval_minutes * 60)
            break
        except asyncio.TimeoutError:
            pass
        try:
            with db_session() as db:
                count = db.execute("SELECT COUNT(*) as c FROM memories WHERE consolidated = 0").fetchone()["c"]
            
            if count >= 2:
                log.info(f"🔄 Running consolidation ({count} unconsolidated memories)...")
                await agent.consolidate()
            else:
                log.debug(f"🔄 Consolidation skipped: {count} memories found (need >= 2)")
        except Exception as e:
            log.error(f"Consolidation error: {e}")

async def deep_reconsolidate_loop(agent: MemoryAgent, interval_hours: int = 24):
    """Run deep re-consolidation every 24 hours."""
    while _shutdown_event and not _shutdown_event.is_set():
        try:
            await asyncio.wait_for(_shutdown_event.wait(), timeout=interval_hours * 3600)
            break
        except asyncio.TimeoutError:
            pass
        try:
            with db_session() as db:
                total = db.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
            
            if total >= 5:
                await agent.deep_reconsolidate()
                await agent.self_improve()
        except Exception as e:
            log.error(f"Deep reconsolidation error: {e}")


# ─── Main ──────────────────────────────────────────────────────

async def main_async(args):
    global _shutdown_event
    _shutdown_event = get_shutdown_event()

    init_db()
    agent = MemoryAgent()

    log.info("🧠 Agent Memory Layer v2 starting (Refactored)")
    
    tasks = [
        asyncio.create_task(watch_folder(agent, Path(args.watch))),
        asyncio.create_task(consolidation_loop(agent, args.consolidate_every)),
        asyncio.create_task(autodream_loop(agent, AUTODREAM_CHECK_INTERVAL)),
        asyncio.create_task(deep_reconsolidate_loop(agent, 24)),
        asyncio.create_task(librarian_loop(on_drift_detected=agent.push_sync_task)),
        asyncio.create_task(agent.sync_worker_loop()),
    ]

    # MCP server (optional — disabled if --mcp-port 0 or MCP_PORT=0)
    from config import MCP_PORT, MCP_HOST
    mcp_port = args.mcp_port if args.mcp_port is not None else MCP_PORT
    mcp_host = args.mcp_host if args.mcp_host is not None else MCP_HOST

    if mcp_port != 0:
        from mcp_server import run_mcp_server
        tasks.append(asyncio.create_task(run_mcp_server(agent, host=mcp_host, port=mcp_port)))
        log.info(f"🔌 MCP server task queued on {mcp_host}:{mcp_port}")
    else:
        log.info("🔌 MCP server disabled (--mcp-port 0 or MCP_PORT=0)")

    app = build_http(agent, watch_path=args.watch)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()

    log.info("✅ Agent running. Press Ctrl+C to stop.")

    try:
        await _shutdown_event.wait()
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await runner.cleanup()
        log.info("🧠 Agent stopped.")

def main():
    parser = argparse.ArgumentParser(description="Agent Memory Layer v2 - PydanticAI")
    parser.add_argument("--watch", default=INBOX_DIR, help="Folder to watch")
    parser.add_argument("--port", type=int, default=8888, help="API port")
    parser.add_argument("--consolidate-every", type=int, default=30, help="Interval in minutes")
    parser.add_argument(
        "--mcp-port",
        type=int,
        default=None,
        help="Port for the MCP server (default: MCP_PORT env var or 8765). Pass 0 to disable MCP.",
    )
    parser.add_argument(
        "--mcp-host",
        type=str,
        default=None,
        help="Bind host for the MCP server (default: MCP_HOST env var or 0.0.0.0).",
    )
    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _signal_handler(sig=None):
        if _shutdown_event:
            _shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        loop.run_until_complete(main_async(args))
    except (KeyboardInterrupt, asyncio.CancelledError):
        if _shutdown_event:
            _shutdown_event.set()
    finally:
        loop.close()

if __name__ == "__main__":
    main()
