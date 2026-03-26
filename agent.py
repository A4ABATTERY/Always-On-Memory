"""
Agent Memory Layer — Always-On PydanticAI Agent Entry Point.

Refactored into a modular architecture for better maintainability and type safety.
"""

import argparse
import asyncio
import logging
import os
import shutil
import signal
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from pydantic_ai import Agent
from aiohttp import web
from config import (
    WATCH_DIRS, IGNORE_DIRS, DB_PATH, MODEL, SMART_MODEL,
    RATE_LIMIT, TEXT_EXTENSIONS, ALL_SUPPORTED, MEDIA_EXTENSIONS,
    _shutdown_event, IDLE_THRESHOLD_MINUTES, AUTODREAM_CHECK_INTERVAL,
    CONSOLIDATION_QUALITY_THRESHOLD
)
from database import init_db, db_session
from agents_factory import build_agents
from utils import retry_with_backoff
from memory_store import (
    get_memory_stats, read_all_memories, delete_memory
)
from librarian import search_documents, librarian_loop
from server import build_http

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
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
        ) = build_agents()
        
        self.client = genai.Client() if HAS_GENAI else None

    async def ingest_file(self, file_path: Path) -> str:
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
        result = await retry_with_backoff(self.ingest_agent.run, msg, shutdown_event=_shutdown_event)
        
        usage = result.usage()
        log.info(f"📥 Ingested: {usage.total_tokens} tokens | {usage.request_tokens} req | {usage.response_tokens} res")
        return result.output

    async def adversarial_consolidation(
        self,
        generator: Agent,
        evaluator: Agent,
        raw_memories_text: str,
        max_attempts: int = 3,
        quality_threshold: float = 0.85,
    ) -> str:
        """
        Generator-Evaluator adversarial loop.
        Reference: Anthropic harness design — "tuning a standalone evaluator to be skeptical turns out to be far more tractable."
        """
        import json
        feedback = "No feedback yet. Generate the initial synthesis."
        
        for attempt in range(max_attempts):
            # Step 1: Generator creates synthesis
            gen_result = await retry_with_backoff(
                generator.run,
                f"Synthesize these memories. Previous feedback: {feedback}\n\nMemories:\n{raw_memories_text}",
                shutdown_event=_shutdown_event,
            )
            draft = gen_result.output
            
            # Step 2: Evaluator grades it
            eval_result = await retry_with_backoff(
                evaluator.run,
                f"Source memories:\n{raw_memories_text}\n\nDraft synthesis:\n{draft}\n\n"
                "Grade strictly. Output JSON ONLY with score, fidelity, completeness, redundancy_removed, feedback.",
                shutdown_event=_shutdown_event,
            )
            
            # Parse evaluation
            try:
                # Basic JSON cleanup in case of markdown wrapping
                eval_text = eval_result.output.strip()
                if eval_text.startswith("```json"):
                    eval_text = eval_text[7:-3].strip()
                elif eval_text.startswith("```"):
                    eval_text = eval_text[3:-3].strip()
                
                eval_data = json.loads(eval_text)
                score = eval_data.get("score", 0.0)
                feedback = eval_data.get("feedback", "No specific feedback.")
            except (json.JSONDecodeError, AttributeError) as e:
                log.warning(f"Evaluator returned non-JSON output, treating as failure: {e}")
                score = 0.0
                feedback = f"Evaluator raw output: {eval_result.output}"
            
            log.info(f"🔄 Consolidation attempt {attempt+1}: score={score:.2f}")
            
            if score >= quality_threshold:
                log.info(f"✅ Consolidation approved (score={score:.2f})")
                return draft
            
            log.info(f"⚠️ Below threshold ({quality_threshold}). Feedback: {feedback[:100]}...")
        
        log.warning(f"❌ Consolidation failed after {max_attempts} attempts. Keeping raw memories.")
        raise Exception("Consolidation quality threshold not met after max attempts.")

    async def consolidate(self) -> str:
        log.info("🔄 Running periodic adversarial consolidation...")
        from memory_store import read_unconsolidated_memories, store_consolidation
        data = read_unconsolidated_memories()
        if data["count"] < 2:
            return "Nothing to consolidate"
        
        import json
        memories_text = json.dumps(data["memories"], indent=2)
        
        try:
            result = await self.adversarial_consolidation(
                self.generator_lite, self.evaluator_lite, memories_text
            )
            return result
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
        file_refs = output.count("/home/") or output.count("./") or output.count("Relevant Files")
        
        usage = result.usage()
        log.info(f"🔍 Answered: {usage.total_tokens} tokens | {memory_refs} refs | {file_refs} files")
        return output

    async def deep_reconsolidate(self) -> str:
        log.info(f"🧠 Running deep adversarial re-consolidation using {SMART_MODEL}...")
        from memory_store import read_all_memories
        data = read_all_memories()
        
        import json
        memories_text = json.dumps(data["memories"], indent=2)
        
        try:
            result = await self.adversarial_consolidation(
                self.generator_smart, self.evaluator_smart, memories_text,
                max_attempts=3, quality_threshold=0.85
            )
            # After consolidation, run self-improvement
            await self.self_improve()
            return result
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
    from config import IDLE_THRESHOLD_MINUTES, CONSOLIDATION_QUALITY_THRESHOLD
    
    log.info(f"💤 AutoDream: checking every {check_interval}s, triggers after {IDLE_THRESHOLD_MINUTES}min idle")
    
    while not _shutdown_event.is_set():
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
    from memory_store import delete_memory
    with db_session() as db:
        rows = db.execute("SELECT id, importance_score, created_at FROM memories WHERE consolidated = 0").fetchall()
        for r in rows:
            created_dt = datetime.fromisoformat(r["created_at"])
            if created_dt.tzinfo is None: created_dt = created_dt.replace(tzinfo=timezone.utc)
            
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
    """Cluster related memories and trigger adversarial consolidation for large clusters."""
    from memory_store import read_unconsolidated_memories
    data = read_unconsolidated_memories(limit=100)
    if data["count"] < 3:
        return

    memories = data["memories"]
    # Simple topic-based clustering
    clusters = {}
    for m in memories:
        # Use the first topic as a cluster key if available
        topics = m.get("topics", [])
        topic = topics[0] if topics else "general"
        if topic not in clusters: clusters[topic] = []
        clusters[topic].append(m)
    
    import json
    for topic, cluster in clusters.items():
        if len(cluster) >= 3:
            log.info(f"💤 Dream: compressing cluster '{topic}' ({len(cluster)} memories)")
            cluster_text = json.dumps(cluster, indent=2)
            try:
                # Use smart agents for Dream consolidation
                await agent.adversarial_consolidation(
                    agent.generator_smart, 
                    agent.evaluator_smart, 
                    cluster_text,
                    quality_threshold=0.9 # Higher quality for dream
                )
            except Exception as e:
                log.debug(f"Dream cluster consolidation failed: {e}")

async def watch_folder(agent: MemoryAgent, folder: Path, poll_interval: int = 5):
    """Watch a folder for new files and ingest them."""
    folder.mkdir(parents=True, exist_ok=True)
    log.info(f"👁️  Watching: {folder}/")

    while not _shutdown_event.is_set():
        try:
            files = sorted(folder.iterdir())
            for f in files:
                if _shutdown_event.is_set():
                    break
                if f.name.startswith("."):
                    continue
                suffix = f.suffix.lower()
                if suffix not in ALL_SUPPORTED:
                    continue
                
                with db_session() as db:
                    row = db.execute("SELECT 1 FROM processed_files WHERE path = ?", (str(f),)).fetchone()
                
                if row:
                    continue

                try:
                    if suffix in TEXT_EXTENSIONS:
                        log.info(f"📄 New text file: {f.name}")
                        text = f.read_text(encoding="utf-8", errors="replace")[:10000]
                        if text.strip():
                            await agent.ingest(text, source=f.name)
                        
                        with db_session() as db:
                            db.execute(
                                "INSERT INTO processed_files (path, processed_at) VALUES (?, ?)",
                                (str(f), datetime.now(timezone.utc).isoformat()),
                            )
                            db.commit()
                    elif suffix in MEDIA_EXTENSIONS:
                        log.info(f"🖼️  New media file: {f.name}")
                        await agent.ingest(f"New media file found: {f.name}", source=f.name)
                        
                        with db_session() as db:
                            db.execute(
                                "INSERT INTO processed_files (path, processed_at) VALUES (?, ?)",
                                (str(f), datetime.now(timezone.utc).isoformat()),
                            )
                            db.commit()
                except Exception as file_err:
                    log.error(f"Error ingesting {f.name}: {file_err}")
                    # Mark as processed even on failure to avoid infinite loop (or use a failure count)
                    with db_session() as db:
                        db.execute(
                            "INSERT INTO processed_files (path, processed_at) VALUES (?, ?)",
                            (str(f), datetime.now(timezone.utc).isoformat() + " (FAILED)"),
                        )
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
    while not _shutdown_event.is_set():
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
    while not _shutdown_event.is_set():
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
    init_db()
    agent = MemoryAgent()

    log.info("🧠 Agent Memory Layer v2 starting (Refactored)")
    
    tasks = [
        asyncio.create_task(watch_folder(agent, Path(args.watch))),
        asyncio.create_task(consolidation_loop(agent, args.consolidate_every)),
        asyncio.create_task(autodream_loop(agent, AUTODREAM_CHECK_INTERVAL)),
        asyncio.create_task(deep_reconsolidate_loop(agent, 24)),
        asyncio.create_task(librarian_loop()),
    ]

    app = build_http(agent, watch_path=args.watch)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()

    log.info(f"✅ Agent running. Press Ctrl+C to stop.")

    try:
        await _shutdown_event.wait()
    finally:
        for t in tasks: t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await runner.cleanup()
        log.info("🧠 Agent stopped.")

def main():
    parser = argparse.ArgumentParser(description="Agent Memory Layer v2 - PydanticAI")
    parser.add_argument("--watch", default="./inbox", help="Folder to watch")
    parser.add_argument("--port", type=int, default=8888, help="API port")
    parser.add_argument("--consolidate-every", type=int, default=30, help="Interval in minutes")
    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _signal_handler(sig=None):
        _shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    try:
        loop.run_until_complete(main_async(args))
    except (KeyboardInterrupt, asyncio.CancelledError):
        _shutdown_event.set()
    finally:
        loop.close()

if __name__ == "__main__":
    main()
