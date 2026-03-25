"""
Agent Memory Layer — Always-On PydanticAI Agent

A lightweight, cost-effective background agent that continuously processes,
consolidates, and serves memory. Runs 24/7 on Gemini 3.1 Flash-Lite.

Features:
    - Memory ingestion, consolidation, and query via PydanticAI agents
    - Temporal knowledge graph (valid_from/valid_to truth windows)
    - Composite scoring (importance × recency) with explainable recall
    - Memory decay loop (activity-aware)
    - Vector search (Librarian mode) via sqlite-vec for source code indexing
    - Smart re-consolidation every 24h via a stronger model
    - Rate limiting via ConcurrencyLimitedModel

Usage:
    python agent.py                          # watch ./inbox, serve on :8888
    python agent.py --watch ./docs --port 9000
    python agent.py --consolidate-every 15   # consolidate every 15 min

Query:
    curl "http://localhost:8888/query?q=what+do+you+know"
    curl -X POST http://localhost:8888/ingest -d '{"text": "some info"}'
"""

import argparse
import asyncio
import contextlib
import hashlib
import json
import logging
import mimetypes
import os
import shutil
import signal
import sqlite3
import struct
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

from aiohttp import web
from pydantic_ai import Agent

try:
    from pydantic_ai import ConcurrencyLimitedModel
except ImportError:
    ConcurrencyLimitedModel = None

try:
    import sqlite_vec
    HAS_SQLITE_VEC = True
except ImportError:
    HAS_SQLITE_VEC = False

try:
    from google import genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# ─── Load .env ─────────────────────────────────────────────────

def _load_dotenv():
    """Load .env file from the script's directory (no external dependency)."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:  # don't override existing env vars
            os.environ[key] = value

_load_dotenv()

# ─── Config ────────────────────────────────────────────────────

MODEL = os.getenv("MODEL", "google-gla:gemini-3.1-flash-lite")
SMART_MODEL = os.getenv("SMART_MODEL", "google-gla:gemini-3.0-flash")
DB_PATH = os.getenv("MEMORY_DB", "memory.db")
RATE_LIMIT = int(os.getenv("RATE_LIMIT", "15"))
WATCH_DIRS = os.getenv("WATCH_DIRS", "")  # comma-separated folder paths
IGNORE_DIRS = os.getenv("IGNORE_DIRS", "")  # comma-separated extra dirs to skip
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2-preview")
SKILLS_DIR = os.getenv("SKILLS_DIR", ".agents/skills")
DEBOUNCE_INTERVAL = int(os.getenv("DEBOUNCE_INTERVAL", "60"))  # seconds to wait after last change before indexing
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "5"))          # seconds between checking for modifications

# Supported file types for multimodal ingestion (inbox watcher)
TEXT_EXTENSIONS = {".txt", ".md", ".json", ".csv", ".log", ".xml", ".yaml", ".yml"}
MEDIA_EXTENSIONS = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
    ".flac": "audio/flac", ".m4a": "audio/mp4", ".aac": "audio/aac",
    ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
    ".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
    ".pdf": "application/pdf",
}
ALL_SUPPORTED = TEXT_EXTENSIONS | set(MEDIA_EXTENSIONS.keys())

# File types for Librarian (vector indexer)
CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".kt",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".sh",
    ".bash", ".zsh", ".sql", ".r", ".scala", ".dart",
    ".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".toml", ".cfg",
    ".ini", ".env", ".dockerfile", ".makefile",
}

# Binary / machine-code extensions to always skip
BINARY_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".bin", ".exe",
    ".o", ".obj", ".a", ".lib", ".class", ".jar", ".war", ".ear",
    ".wasm", ".whl", ".egg", ".zip", ".tar", ".gz", ".bz2", ".xz",
    ".7z", ".rar", ".iso", ".dmg", ".deb", ".rpm",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv", ".flac",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".sqlite", ".db", ".sqlite3", ".lock",
    ".min.js", ".min.css", ".map",
}

# Directories to always skip during indexing
SKIP_DIRS = {
    "__pycache__", "node_modules", ".git", ".hg", ".svn",
    "venv", ".venv", "env", ".env", ".tox", ".mypy_cache",
    ".pytest_cache", "dist", "build", ".eggs", "*.egg-info",
    ".next", ".nuxt", "coverage", ".coverage",
    "vendor", "target",  # Go/Rust build dirs
}


def _get_latest_mtime(dirs: list[str]) -> float:
    """Find the maximum modification time across all relevant files in the watched directories."""
    extra_ignores = {d.strip() for d in IGNORE_DIRS.split(",") if d.strip()}
    all_skip = SKIP_DIRS | extra_ignores

    latest_mtime = 0.0
    for dir_path in dirs:
        folder = Path(dir_path)
        if not folder.is_dir():
            continue
        
        for f in folder.rglob("*"):
            if not f.is_file() or f.suffix.lower() not in CODE_EXTENSIONS:
                continue
            
            parts = f.parts
            if any(p.startswith(".") or p in all_skip for p in parts):
                continue
            
            try:
                mtime = f.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
            except (OSError, ValueError):
                continue
    return latest_mtime


def is_binary_file(file_path: Path) -> bool:
    """Quick heuristic: check if a file is binary by reading its first 8KB."""
    if file_path.suffix.lower() in BINARY_EXTENSIONS:
        return True
    try:
        chunk = file_path.read_bytes()[:8192]
        # If more than 10% of bytes are non-text, it's binary
        non_text = sum(1 for b in chunk if b < 8 or (14 <= b < 32 and b != 27))
        return (non_text / max(len(chunk), 1)) > 0.10
    except Exception:
        return True


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="[%H:%M]",
)
log = logging.getLogger("memory-agent")


# ─── Rate-Limited Models ───────────────────────────────────────

def make_model(model_str: str):
    """Wrap a model string with ConcurrencyLimitedModel if available."""
    if ConcurrencyLimitedModel and RATE_LIMIT > 0:
        return ConcurrencyLimitedModel(model_str, limiter=RATE_LIMIT)
    return model_str

lite_model = make_model(MODEL)
smart_model = make_model(SMART_MODEL)


# ─── Database ──────────────────────────────────────────────────


def get_db() -> sqlite3.Connection:
    """Returns a raw connection (for legacy calls). Use db_session instead."""
    db = sqlite3.connect(DB_PATH, timeout=30.0)
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute("PRAGMA busy_timeout=5000")
    except Exception:
        pass
    if HAS_SQLITE_VEC:
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
    return db


@contextlib.contextmanager
def db_session():
    """Context manager that ensures the database connection is closed."""
    db = get_db()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initialize database schema and run migrations."""
    with db_session() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL DEFAULT '',
                raw_text TEXT NOT NULL,
                summary TEXT NOT NULL,
                entities TEXT NOT NULL DEFAULT '[]',
                topics TEXT NOT NULL DEFAULT '[]',
                connections TEXT NOT NULL DEFAULT '[]',
                importance REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                consolidated INTEGER NOT NULL DEFAULT 0,
                sector TEXT NOT NULL DEFAULT 'semantic',
                valid_to TEXT DEFAULT NULL
            );
            CREATE TABLE IF NOT EXISTS consolidations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_ids TEXT NOT NULL,
                summary TEXT NOT NULL,
                insight TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS processed_files (
                path TEXT PRIMARY KEY,
                processed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                UNIQUE(path, chunk_index)
            );
        """)

        # Create vec0 virtual table for vector search (if sqlite-vec loaded)
        if HAS_SQLITE_VEC:
            try:
                # Check if vec_documents already exists
                db.execute("SELECT 1 FROM vec_documents LIMIT 1").fetchone()
            except sqlite3.OperationalError:
                try:
                    db.execute("""
                        CREATE VIRTUAL TABLE vec_documents USING vec0(
                            document_id INTEGER PRIMARY KEY,
                            embedding float[3072]
                        )
                    """)
                except Exception as e:
                    log.warning(f"Could not create vec_documents: {e}")

        # Migrations for existing DB
        for col_sql in [
            "ALTER TABLE memories ADD COLUMN sector TEXT NOT NULL DEFAULT 'semantic'",
            "ALTER TABLE memories ADD COLUMN valid_to TEXT DEFAULT NULL",
        ]:
            try:
                db.execute(col_sql)
            except sqlite3.OperationalError:
                pass


# Initialize at startup
init_db()


# ─── Embedding Helper ─────────────────────────────────────────

def serialize_f32(vector: list[float]) -> bytes:
    """Serialize a list of floats into compact binary format for sqlite-vec."""
    return struct.pack("%sf" % len(vector), *vector)


# Rate limiter for embedding calls — controls how fast we hit the embedding API
_embed_semaphore = asyncio.Semaphore(4)  # max 3 concurrent embed calls
_EMBED_DELAY = 1.0 / max(RATE_LIMIT, 4)  # seconds between embed calls


async def embed_text(text: str) -> list[float] | None:
    """Generate an embedding for the given text using google-genai (rate-limited)."""
    if not HAS_GENAI:
        return None
    if not text or not text.strip():
        return None
    if _shutdown_event.is_set():
        return None
    
    # We use a global/singleton client if possible, but for reliability 
    # and simplicity here, we use genai.Client() which is fast.
    # In the MemoryAgent class, we'll store a reference to the client.
    async def _do_embed():
        client = genai.Client()
        result = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text[:2000],  # truncate to stay within limits
        )
        return result.embeddings[0].values

    async with _embed_semaphore:
        try:
            embedding = await retry_with_backoff(_do_embed)
            await asyncio.sleep(_EMBED_DELAY)  # rate limit: space out requests
            return embedding
        except Exception as e:
            log.error(f"Embedding error: {e}")
            return None


# ─── Memory Tool Functions ─────────────────────────────────────


def store_memory(
    raw_text: str,
    summary: str,
    entities: list[str],
    topics: list[str],
    importance: float,
    sector: str = "semantic",
    valid_to: str | None = None,
    source: str = "",
) -> dict:
    """Store a processed memory in the database.

    Args:
        raw_text: The original input text.
        summary: A concise 1-2 sentence summary.
        entities: Key people, companies, products, or concepts.
        topics: 2-4 topic tags.
        importance: Float 0.0 to 1.0 indicating importance.
        sector: Memory sector: semantic, episodic, procedural, or reflection.
        valid_to: Optional ISO datetime when this fact expires.
        source: Where this memory came from (filename, URL, etc).

    Returns:
        dict with memory_id and confirmation.
    """
    with db_session() as db:
        now = datetime.now(timezone.utc).isoformat()
        cursor = db.execute(
            """INSERT INTO memories (source, raw_text, summary, entities, topics, importance, created_at, sector, valid_to)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (source, raw_text, summary, json.dumps(entities), json.dumps(topics), importance, now, sector, valid_to),
        )
        db.commit()
        mid = cursor.lastrowid
    
    log.info(f"📥 Stored memory #{mid}: {summary[:60]}...")
    return {"memory_id": mid, "status": "stored", "summary": summary}


def read_all_memories() -> dict:
    """Read and rank stored memories based on composite score (recency, importance).

    Returns:
        dict with list of memories and count.
    """
    with db_session() as db:
        now_iso = datetime.now(timezone.utc).isoformat()
        now_ts = datetime.now(timezone.utc).timestamp()

        rows = db.execute(
            "SELECT * FROM memories WHERE valid_to IS NULL OR valid_to > ? ORDER BY created_at DESC LIMIT 200",
            (now_iso,)
        ).fetchall()

    memories = []
    for r in rows:
        created_ts = datetime.fromisoformat(r["created_at"]).replace(tzinfo=timezone.utc).timestamp()
        age_hours = (now_ts - created_ts) / 3600.0
        score = r["importance"] * (1.0 / (1.0 + (age_hours / 24.0)))

        memories.append({
            "id": r["id"], "source": r["source"], "summary": r["summary"],
            "entities": json.loads(r["entities"]), "topics": json.loads(r["topics"]),
            "importance": r["importance"], "connections": json.loads(r["connections"]),
            "created_at": r["created_at"], "consolidated": bool(r["consolidated"]),
            "sector": r["sector"], "valid_to": r["valid_to"],
            "composite_score": score,
            "recall_reason": [
                "High Importance" if r["importance"] > 0.7 else "Standard",
                f"Age: {age_hours:.1f}h",
                f"Score: {score:.2f}"
            ]
        })

    memories.sort(key=lambda x: x["composite_score"], reverse=True)
    memories = memories[:50]
    return {"memories": memories, "count": len(memories)}


def read_unconsolidated_memories() -> dict:
    """Read memories that haven't been consolidated yet.

    Returns:
        dict with list of unconsolidated memories and count.
    """
    with db_session() as db:
        rows = db.execute(
            "SELECT * FROM memories WHERE consolidated = 0 ORDER BY created_at DESC LIMIT 30"
        ).fetchall()
    memories = []
    for r in rows:
        memories.append({
            "id": r["id"], "summary": r["summary"],
            "entities": json.loads(r["entities"]), "topics": json.loads(r["topics"]),
            "importance": r["importance"], "created_at": r["created_at"],
        })
    return {"memories": memories, "count": len(memories)}


def read_memory_partition(sector: str) -> dict:
    """Fetches memories only from a specific sector (semantic, episodic, procedural, reflection).

    Args:
        sector: The sector name to filter by.
    """
    with db_session() as db:
        rows = db.execute(
            "SELECT * FROM memories WHERE sector = ? ORDER BY created_at DESC LIMIT 100",
            (sector,)
        ).fetchall()
    memories = []
    for r in rows:
        memories.append({
            "id": r["id"], "summary": r["summary"], "raw_text": r["raw_text"],
            "importance": r["importance"], "created_at": r["created_at"],
        })
    return {"sector": sector, "memories": memories, "count": len(memories)}


def write_skill_file(skill_name: str, content: str) -> dict:
    """Writes a new skill or rule to the configured SKILLS_DIR.

    Args:
        skill_name: Short name for the skill (used for filename).
        content: The full markdown content of the SKILL.md file.
    """
    clean_name = "".join(c for c in skill_name if c.isalnum() or c in ("-", "_")).lower()
    target_dir = Path(SKILLS_DIR) / clean_name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    skill_path = target_dir / "SKILL.md"
    skill_path.write_text(content, encoding="utf-8")
    
    log.info(f"💾 Saved skill: {clean_name} to {skill_path}")
    return {"status": "saved", "path": str(skill_path), "skill_name": clean_name}


def store_consolidation(
    source_ids: list[int],
    summary: str,
    insight: str,
    connections: list[dict],
) -> dict:
    """Store a consolidation result and mark source memories as consolidated.

    Args:
        source_ids: List of memory IDs that were consolidated.
        summary: A synthesized summary across all source memories.
        insight: One key pattern or insight discovered.
        connections: List of dicts with 'from_id', 'to_id', 'relationship'.

    Returns:
        dict with confirmation.
    """
    with db_session() as db:
        now = datetime.now(timezone.utc).isoformat()
        db.execute(
            "INSERT INTO consolidations (source_ids, summary, insight, created_at) VALUES (?, ?, ?, ?)",
            (json.dumps(source_ids), summary, insight, now),
        )
        for conn in connections:
            from_id, to_id = conn.get("from_id"), conn.get("to_id")
            rel = conn.get("relationship", "")
            if from_id and to_id:
                for mid in [from_id, to_id]:
                    row = db.execute("SELECT connections FROM memories WHERE id = ?", (mid,)).fetchone()
                    if row:
                        existing = json.loads(row["connections"])
                        existing.append({"linked_to": to_id if mid == from_id else from_id, "relationship": rel})
                        db.execute("UPDATE memories SET connections = ? WHERE id = ?", (json.dumps(existing), mid))
        placeholders = ",".join("?" * len(source_ids))
        db.execute(f"UPDATE memories SET consolidated = 1 WHERE id IN ({placeholders})", source_ids)
        db.commit()
    
    log.info(f"🔄 Consolidated {len(source_ids)} memories. Insight: {insight[:80]}...")
    return {"status": "consolidated", "memories_processed": len(source_ids), "insight": insight}


def read_consolidation_history() -> dict:
    """Read past consolidation insights.

    Returns:
        dict with list of consolidation records.
    """
    with db_session() as db:
        rows = db.execute("SELECT * FROM consolidations ORDER BY created_at DESC LIMIT 10").fetchall()
        result = [{"summary": r["summary"], "insight": r["insight"], "source_ids": r["source_ids"]} for r in rows]
    return {"consolidations": result, "count": len(result)}


def get_memory_stats() -> dict:
    """Get current memory statistics.

    Returns:
        dict with counts of memories, consolidations, indexed documents, etc.
    """
    with db_session() as db:
        total = db.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
        unconsolidated = db.execute("SELECT COUNT(*) as c FROM memories WHERE consolidated = 0").fetchone()["c"]
        consolidations = db.execute("SELECT COUNT(*) as c FROM consolidations").fetchone()["c"]
        indexed_docs = db.execute("SELECT COUNT(DISTINCT path) as c FROM documents").fetchone()["c"]
    
    return {
        "total_memories": total,
        "unconsolidated": unconsolidated,
        "consolidations": consolidations,
        "indexed_documents": indexed_docs,
    }


def delete_memory(memory_id: int) -> dict:
    """Delete a memory by ID.

    Args:
        memory_id: The ID of the memory to delete.

    Returns:
        dict with status.
    """
    with db_session() as db:
        row = db.execute("SELECT 1 FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            return {"status": "not_found", "memory_id": memory_id}
        db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        db.commit()
    
    log.info(f"🗑️  Deleted memory #{memory_id}")
    return {"status": "deleted", "memory_id": memory_id}


def clear_all_memories(inbox_path: str | None = None) -> dict:
    """Delete all memories, consolidations, and inbox files. Full reset."""
    with db_session() as db:
        mem_count = db.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
        db.execute("DELETE FROM memories")
        db.execute("DELETE FROM consolidations")
        db.execute("DELETE FROM processed_files")
        db.commit()

    files_deleted = 0
    if inbox_path:
        folder = Path(inbox_path)
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
    return {"status": "cleared", "memories_deleted": mem_count, "files_deleted": files_deleted}


def update_memory_validity(memory_id: int, valid_to: str) -> dict:
    """Set the valid_to timestamp for an existing memory to mark it as obsolete or expired.

    Args:
        memory_id: ID of the memory to update.
        valid_to: ISO format datetime string (e.g. '2026-03-24T00:00:00')
    """
    with db_session() as db:
        db.execute("UPDATE memories SET valid_to = ? WHERE id = ?", (valid_to, memory_id))
        db.commit()
    log.info(f"⏱️ Updated validity of memory #{memory_id} to {valid_to}")
    return {"status": "updated", "memory_id": memory_id, "valid_to": valid_to}


def reinforce_memory(memory_id: int) -> dict:
    """Increase the importance of a memory and reset its decay clock because it was reinforced by new information.

    Args:
        memory_id: ID of the memory to reinforce.
    """
    with db_session() as db:
        row = db.execute("SELECT importance FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            return {"status": "not_found"}

        new_importance = min(1.0, row["importance"] + 0.1)
        now = datetime.now(timezone.utc).isoformat()

        db.execute("UPDATE memories SET importance = ?, created_at = ? WHERE id = ?", (new_importance, now, memory_id))
        db.commit()
    log.info(f"💪 Reinforced memory #{memory_id} (Importance: {row['importance']:.2f} -> {new_importance:.2f})")
    return {"status": "reinforced", "memory_id": memory_id, "new_importance": new_importance}


async def search_documents(query: str, k: int = 5) -> dict:
    """Search indexed source code and documents by semantic similarity (async).

    Args:
        query: The search query text.
        k: Number of results to return.

    Returns:
        dict with list of matching file paths and snippets.
    """
    if not HAS_SQLITE_VEC or not HAS_GENAI:
        return {"results": [], "error": "Vector search not available (missing sqlite-vec or google-genai)"}

    # Generate embedding using the rate-limited helper
    query_embedding = await embed_text(query)
    if not query_embedding:
        return {"results": [], "error": "Failed to generate embedding for query."}

    with db_session() as db:
        rows = db.execute(
            """
            SELECT v.document_id, v.distance, d.path, d.chunk_text, d.chunk_index
            FROM vec_documents v
            JOIN documents d ON d.id = v.document_id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
            """,
            [serialize_f32(query_embedding), k],
        ).fetchall()

        results = []
        seen_paths = set()
        for r in rows:
            path = r["path"]
            if path not in seen_paths:
                results.append({
                    "path": path,
                    "snippet": r["chunk_text"][:200],
                    "distance": r["distance"],
                    "chunk_index": r["chunk_index"],
                })
        return {"results": results, "count": len(results)}


def read_document(path: str) -> dict:
    """Read the full content of a document/file from disk (safe).
    Only allows reading from WATCH_DIRS or the inbox.

    Args:
        path: Absolute or relative path to the file.
    """
    file_path = Path(path).resolve()
    
    # Security: check if path is within allowed directories
    allowed_dirs = [Path("inbox").resolve(), Path(SKILLS_DIR).resolve()]
    if WATCH_DIRS:
        allowed_dirs.extend([Path(d.strip()).resolve() for d in WATCH_DIRS.split(",") if d.strip()])
    
    is_allowed = any(str(file_path).startswith(str(d)) for d in allowed_dirs)
    if not is_allowed:
        return {"error": f"Access denied: {path} is not in a watched directory."}
    
    if not file_path.exists():
        return {"error": f"File not found: {path}"}
    if not file_path.is_file():
        return {"error": f"Not a file: {path}"}
    if is_binary_file(file_path):
        return {"error": f"Cannot read binary file: {path}"}

    try:
        content = file_path.read_text(encoding="utf-8", errors="replace")
        return {
            "path": str(file_path),
            "content": content[:20000],  # cap at 20k chars
            "length": len(content),
            "truncated": len(content) > 20000
        }
    except Exception as e:
        return {"error": f"Read error: {e}"}


# ─── PydanticAI Agents ─────────────────────────────────────────

INGEST_SYSTEM_PROMPT = (
    "You are a Memory Ingest Agent. You handle ALL types of input — text, images, "
    "audio, video, and PDFs. For any input you receive:\n"
    "1. Thoroughly describe what the content contains\n"
    "2. Create a concise 1-2 sentence summary\n"
    "3. Extract key entities (people, companies, products, concepts, objects, locations)\n"
    "4. Assign 2-4 topic tags\n"
    "5. Rate importance from 0.0 to 1.0\n"
    "6. Identify the memory sector (semantic, episodic, procedural, or reflection)\n"
    "7. If the fact has an expiration, provide valid_to (ISO datetime string)\n"
    "8. Call store_memory with all extracted information\n\n"
    "For images: describe the scene, objects, text, people, and any visual details.\n"
    "For audio/video: describe the spoken content, sounds, scenes, and key moments.\n"
    "For PDFs: extract and summarize the document content.\n\n"
    "Use the full description as raw_text in store_memory so the context is preserved.\n"
    "Always call store_memory. Be Detailed and accurate.\n"
    "After storing, confirm what was stored in one sentence."
)

CONSOLIDATE_SYSTEM_PROMPT = (
    "You are a Memory Consolidation Agent. You:\n"
    "1. Call read_unconsolidated_memories to see what needs processing\n"
    "2. If fewer than 2 memories, say nothing to consolidate\n"
    "3. Find connections and patterns across the memories.\n"
    "4. Create synthesized summaries and insights. PERFORM MULTIPLE store_consolidation "
    "calls if memories cover disparate themes (e.g. don't mix UI and Database).\n"
    "5. Call store_consolidation with source_ids, summary, insight, and connections\n"
    "6. Contradictions: Call update_memory_validity for old memories if a newer memory contradicts it.\n"
    "7. Reinforcement: Call reinforce_memory for old memories supported by new info.\n\n"
    "Connections: list of dicts with 'from_id', 'to_id', 'relationship' keys.\n"
    "Prioritize thematic clustering over broad summarization."
)

QUERY_SYSTEM_PROMPT = (
    "You are a Memory Query Agent. When asked a question:\n"
    "1. Call read_all_memories to access the memory store\n"
    "2. Call read_consolidation_history for higher-level insights\n"
    "3. Call search_documents to find relevant source code files or documents\n"
    "4. Synthesize an answer based ONLY on stored memories\n"
    "5. Reference memory IDs: [Memory 1], [Memory 2], etc.\n"
    "6. Include a 'Relevant Files' section listing file paths from search_documents results\n"
    "7. If no relevant memories exist, say so honestly\n\n"
    "Be thorough. Always cite sources."
)

DEEP_CONSOLIDATE_SYSTEM_PROMPT = (
    "You are a Deep Memory Consolidation Agent. You are the HIGH-FIDELITY CORRECTIVE layer.\n"
    "Your job is to catch details the 'lite' agent missed or over-summarized.\n"
    "1. Call read_all_memories to see the full context\n"
    "2. Call search_documents and read_document to verify facts against source code/docs\n"
    "3. Directly link insights to relevant file paths in your summary/insight text\n"
    "4. Look for deep architectural patterns, contradictions, and themes\n"
    "5. Reinforce still-relevant memories (reinforce_memory)\n"
    "6. Mark outdated facts as invalid (update_memory_validity)\n"
    "7. PERFORM MULTIPLE store_consolidation calls for different high-level themes.\n\n"
    "Be precise and analytical. Link facts to files."
)

SELF_IMPROVEMENT_SYSTEM_PROMPT = (
    "You are a Self-Improvement Agent. Your goal is to evolve the project's capabilities "
    "by discovering and refining skills based on past performance and failures.\n\n"
    "1. Call read_memory_partition for 'reflection' and 'episodic' sectors.\n"
    "2. Identify recurring failure patterns (EvoSkill thresholds: ≥3 errors, ≥2 hallucinations).\n"
    "3. Identify successful complex workflows that should be codified.\n"
    "4. Use Anthropic's Skill-Creator patterns to write new SKILL.md files:\n"
    "   - Name: Short, descriptive ID.\n"
    "   - Description: When to trigger (MAKE IT PUSHY to avoid undertriggering).\n"
    "   - Instructions: Imperative, clear steps, examples of input/output.\n"
    "5. Use EvoSkill taxonomy: Procedural guide, Scoped constraint, Correction reference, Meta-strategy, Style guide.\n"
    "6. Call write_skill_file to persist the new or updated skill.\n"
    "7. Call search_documents to see if a similar skill already exists before creating a new one.\n\n"
    "Be proactive. If you see a way to make the Orchestrator more reliable, codify it as a skill."
)


def build_agents():
    """Build PydanticAI agents for ingest, consolidate, query, and self-improvement."""
    ingest_agent = Agent(
        lite_model,
        system_prompt=INGEST_SYSTEM_PROMPT,
        tools=[store_memory],
    )

    consolidate_agent = Agent(
        lite_model,
        system_prompt=CONSOLIDATE_SYSTEM_PROMPT,
        tools=[read_unconsolidated_memories, store_consolidation, update_memory_validity, reinforce_memory],
    )

    query_agent = Agent(
        lite_model,
        system_prompt=QUERY_SYSTEM_PROMPT,
        tools=[read_all_memories, read_consolidation_history, search_documents],
    )

    deep_consolidate_agent = Agent(
        smart_model,
        system_prompt=DEEP_CONSOLIDATE_SYSTEM_PROMPT,
        tools=[
            read_all_memories, read_consolidation_history, 
            update_memory_validity, reinforce_memory,
            search_documents, read_document, store_consolidation
        ],
    )

    self_improvement_agent = Agent(
        smart_model,
        system_prompt=SELF_IMPROVEMENT_SYSTEM_PROMPT,
        tools=[
            read_memory_partition, search_documents, 
            read_document, write_skill_file
        ],
    )

    return ingest_agent, consolidate_agent, query_agent, deep_consolidate_agent, self_improvement_agent


# ─── Retry with Backoff ────────────────────────────────────────


async def retry_with_backoff(coro_fn, *args, max_retries: int = 5, base_delay: float = 3.0, **kwargs):
    """Retry an async function with exponential backoff on 429 and 503 errors.
    
    503 (Unavailable) errors are common with high-demand models.
    """
    last_error = None
    delay = base_delay
    
    for attempt in range(max_retries + 1):
        try:
            return await coro_fn(*args, **kwargs)
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            
            # Check for retryable conditions: 429 (Rate Limit), 503 (Unavailable), or specific keywords
            is_retryable = any(k in error_str for k in ["429", "503", "unavailable", "quota", "resource_exhausted", "high demand"])
            
            if not is_retryable or attempt == max_retries:
                log.error(f"❌ Operation failed after {attempt} attempts: {e}")
                raise e
            
            # 503s often need longer wait than 429s (adaptive delay)
            actual_delay = delay * (2.0 if "503" in error_str or "unavailable" in error_str else 1.0)
            log.warning(f"⚠️ Attempt {attempt+1} failed: {e}. Retrying in {actual_delay:.1f}s...")
            
            try:
                await asyncio.wait_for(_shutdown_event.wait(), timeout=actual_delay)
                # If shutdown was set during wait, we exit early
                if _shutdown_event.is_set():
                    return None
            except asyncio.TimeoutError:
                pass
            
            delay *= 2
    raise last_error


# ─── Agent Runner ──────────────────────────────────────────────


class MemoryAgent:
    def __init__(self):
        (
            self.ingest_agent,
            self.consolidate_agent,
            self.query_agent,
            self.deep_consolidate_agent,
            self.self_improvement_agent,
        ) = build_agents()
        if HAS_GENAI:
            self.client = genai.Client()
        else:
            self.client = None

    async def ingest(self, text: str, source: str = "") -> str:
        log.info(f"📥 Analyzing {'file: ' + source if source else 'text content'} ({len(text)} chars)...")
        msg = f"Remember this information (source: {source}):\n\n{text}" if source else f"Remember this information:\n\n{text}"
        result = await retry_with_backoff(self.ingest_agent.run, msg)
        
        # Log token usage
        usage = result.usage()
        log.info(f"📥 Ingested: {usage.total_tokens} tokens total (Prompt: {usage.request_tokens}, Response: {usage.response_tokens})")
        
        return result.output

    async def consolidate(self) -> str:
        log.info("🔄 Running periodic consolidation...")
        result = await retry_with_backoff(
            self.consolidate_agent.run,
            "Consolidate unconsolidated memories. Find connections and patterns.",
        )
        usage = result.usage()
        log.info(f"🔄 Consolidation complete: {usage.total_tokens} tokens used")
        return result.output

    async def query(self, question: str) -> str:
        log.info(f"🔍 Processing query: '{question}'")
        result = await retry_with_backoff(
            self.query_agent.run,
            f"Based on my memories, answer: {question}",
        )
        
        # Parse result for citations and file connections
        output = result.output
        memory_refs = output.count("[Memory")
        file_refs = output.count("/home/") or output.count("./") or output.count("Relevant Files")
        
        usage = result.usage()
        log.info(f"🔍 Answered: {usage.total_tokens} tokens | {memory_refs} memory citations | {file_refs} file references")
        
        return output

    async def deep_reconsolidate(self) -> str:
        log.info(f"🧠 Running deep re-consolidation using {SMART_MODEL}...")
        result = await retry_with_backoff(
            self.deep_consolidate_agent.run,
            "Perform a deep review and re-consolidation of ALL memories. "
            "Find patterns, close outdated truths, reinforce current knowledge.",
        )
        usage = result.usage()
        log.info(f"🧠 Deep re-consolidation complete: {usage.total_tokens} tokens used")
        return result.output

    async def self_improve(self) -> str:
        log.info(f"🧬 Running self-improvement audit using {SMART_MODEL}...")
        result = await retry_with_backoff(
            self.self_improvement_agent.run,
            "Audit recent reflection and episodic memories to discover or refine skills. "
            f"Target directory: {SKILLS_DIR}",
        )
        usage = result.usage()
        log.info(f"🧬 Self-improvement audit complete: {usage.total_tokens} tokens used")
        return result.output

    async def status(self) -> dict:
        return get_memory_stats()

    async def ingest_file(self, file_path: Path) -> str:
        """Ingest a text-based file from the inbox."""
        suffix = file_path.suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            text = file_path.read_text(encoding="utf-8", errors="replace")[:10000]
            if text.strip():
                return await self.ingest(text, source=file_path.name)
        return f"Skipped: unsupported or empty file {file_path.name}"


# ─── File Watcher ──────────────────────────────────────────────


async def watch_folder(agent: MemoryAgent, folder: Path, poll_interval: int = 5):
    """Watch a folder for new files and ingest them."""
    folder.mkdir(parents=True, exist_ok=True)
    log.info(f"👁️  Watching: {folder}/")

    while not _shutdown_event.is_set():
        try:
            # Refresh file list each poll
            files = sorted(folder.iterdir())
            for f in files:
                if _shutdown_event.is_set():
                    break
                if f.name.startswith("."):
                    continue
                suffix = f.suffix.lower()
                if suffix not in ALL_SUPPORTED:
                    continue
                
                # Check if already processed (short-lived connection)
                with db_session() as db:
                    row = db.execute("SELECT 1 FROM processed_files WHERE path = ?", (str(f),)).fetchone()
                
                if row:
                    continue

                try:
                    if suffix in TEXT_EXTENSIONS:
                        log.info(f"📄 New text file: {f.name}")
                        text = f.read_text(encoding="utf-8", errors="replace")[:10000]
                        if text.strip():
                            # CALL LLM (Connection is CLOSED here)
                            await agent.ingest(text, source=f.name)
                        
                        # Mark as processed (short-lived connection)
                        with db_session() as db:
                            db.execute(
                                "INSERT INTO processed_files (path, processed_at) VALUES (?, ?)",
                                (str(f), datetime.now(timezone.utc).isoformat()),
                            )
                            db.commit()
                    elif suffix in MEDIA_EXTENSIONS:
                        log.info(f"🖼️  New media file: {f.name}")
                        # Ingest media (the ingest agent prompt handles describing them)
                        # We pass the file path as text context, but genai will need the actual bytes
                        # The ingest agent needs to be able to handle media.
                        # For now, we'll read text if possible, but the plan was to align.
                        # To truly support media in PydanticAI with genai, we'd need to pass parts.
                        # For now, we'll at least not skip it if it's supported.
                        await agent.ingest(f"New media file found: {f.name}", source=f.name)
                        
                        # Mark as processed
                        with db_session() as db:
                            db.execute(
                                "INSERT INTO processed_files (path, processed_at) VALUES (?, ?)",
                                (str(f), datetime.now(timezone.utc).isoformat()),
                            )
                            db.commit()
                except Exception as file_err:
                    log.error(f"Error ingesting {f.name}: {file_err}")

        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Watch error: {e}")

        try:
            await asyncio.wait_for(_shutdown_event.wait(), timeout=poll_interval)
            break
        except asyncio.TimeoutError:
            pass


# ─── Document Indexer (Librarian Mode) ─────────────────────────


async def index_documents_loop(interval_minutes: int = 60):
    """Periodically index source code and documents from WATCH_DIRS for vector search."""
    if not WATCH_DIRS:
        log.info("📚 Librarian mode disabled (WATCH_DIRS not set)")
        return
    if not HAS_SQLITE_VEC:
        log.warning("📚 Librarian mode disabled (sqlite-vec not installed)")
        return
    if not HAS_GENAI:
        log.warning("📚 Librarian mode disabled (google-genai not installed)")
        return

    dirs = [d.strip() for d in WATCH_DIRS.split(",") if d.strip()]
    log.info(f"📚 Librarian: monitoring {len(dirs)} folder(s) for changes (debounce: {DEBOUNCE_INTERVAL}s)")

    # 0. Get initial last indexed time from DB
    with db_session() as db:
        row = db.execute("SELECT MAX(updated_at) as last_idx FROM documents").fetchone()
        last_indexed_iso = row["last_idx"] if row and row["last_idx"] else "1970-01-01T00:00:00"
        last_indexed_time = datetime.fromisoformat(last_indexed_iso).replace(tzinfo=timezone.utc).timestamp()

    last_change_time = None
    current_max_mtime = last_indexed_time
    
    while not _shutdown_event.is_set():
        try:
            # 1. Get latest modification time across all files
            latest_mtime = _get_latest_mtime(dirs)
            
            # 2. If we find a file newer than our current max, it's a new change
            if latest_mtime > current_max_mtime:
                if last_change_time is None:
                    log.info(f"📚 Librarian: modification detected. Starting {DEBOUNCE_INTERVAL}s debounce...")
                else:
                    log.info(f"📚 Librarian: further modification detected. Resetting {DEBOUNCE_INTERVAL}s timer.")
                last_change_time = time.time()
                current_max_mtime = latest_mtime
            
            # 3. If debounce period has passed, run full indexing
            if last_change_time and (time.time() - last_change_time >= DEBOUNCE_INTERVAL):
                log.info("📚 Librarian: debounce window closed. Synchronizing vector index...")
                await _index_all_dirs(dirs)
                
                # Refresh last_indexed_time from DB
                with db_session() as db:
                    row = db.execute("SELECT MAX(updated_at) as last_idx FROM documents").fetchone()
                    if row and row["last_idx"]:
                        last_indexed_iso = row["last_idx"]
                        last_indexed_time = datetime.fromisoformat(last_indexed_iso).replace(tzinfo=timezone.utc).timestamp()
                
                last_change_time = None
                current_max_mtime = last_indexed_time
                
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.error(f"Indexing error: {e}")
        # Poll internal wait
        try:
            await asyncio.wait_for(_shutdown_event.wait(), timeout=SCAN_INTERVAL)
            break
        except asyncio.TimeoutError:
            pass


async def _index_all_dirs(dirs: list[str]):
    """Walk directories and index files."""
    indexed = 0
    skipped = 0

    for dir_path in dirs:
        folder = Path(dir_path)
        if not folder.is_dir():
            log.warning(f"📚 Skipping non-existent directory: {dir_path}")
            continue

        # Pre-compute ignore set once per run
        extra_ignores = {d.strip() for d in IGNORE_DIRS.split(",") if d.strip()}
        all_skip = SKIP_DIRS | extra_ignores

        for f in folder.rglob("*"):
            # ─── Shutdown check on every file ───
            if _shutdown_event.is_set():
                log.info("📚 Indexing interrupted by shutdown.")
                return

            if not f.is_file():
                continue
            if f.suffix.lower() not in CODE_EXTENSIONS:
                continue
            
            parts = f.parts
            if any(p.startswith(".") or p in all_skip for p in parts):
                continue
            if is_binary_file(f):
                continue

            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            content_hash = hashlib.md5(text.encode()).hexdigest()

            # Check existing (short-lived connection)
            with db_session() as db:
                existing = db.execute(
                    "SELECT content_hash FROM documents WHERE path = ? AND chunk_index = 0",
                    (str(f),)
                ).fetchone()

                if existing and existing["content_hash"] == content_hash:
                    skipped += 1
                    continue

                # Delete old chunks
                db.execute("DELETE FROM documents WHERE path = ?", (str(f),))
                db.commit()

            # Chunk the file (LLM call ahead)
            chunks = _chunk_text(text, max_chars=1500)
            log.info(f"📚 Indexing '{f.name}': {len(chunks)} chunks...")
            now = datetime.now(timezone.utc).isoformat()

            for i, chunk in enumerate(chunks):
                if _shutdown_event.is_set():
                    break

                # Generate embedding (Connection CLOSED here)
                embedding = await embed_text(chunk)
                
                # Store (short-lived connection)
                with db_session() as db: 
                    cursor = db.execute(
                        "INSERT INTO documents (path, content_hash, chunk_text, chunk_index, updated_at) VALUES (?, ?, ?, ?, ?)",
                        (str(f), content_hash, chunk, i, now),
                    )
                    doc_id = cursor.lastrowid
                    
                    if embedding:
                        try:
                            # Vec index relies on the same session
                            db.execute(
                                "INSERT INTO vec_documents (document_id, embedding) VALUES (?, ?)",
                                (doc_id, serialize_f32(embedding)),
                            )
                        except Exception as e:
                            log.error(f"Vec insert error for {f.name}: {e}")
                    db.commit()

            indexed += 1

    if indexed > 0 or skipped > 0:
        log.info(f"📚 Indexing complete: {indexed} files indexed, {skipped} unchanged")


def _chunk_text(text: str, max_chars: int = 1500) -> list[str]:
    """Split text into chunks, trying to break on newlines."""
    if len(text) <= max_chars:
        return [text]

    chunks = []
    lines = text.split("\n")
    current_chunk = ""

    for line in lines:
        if len(current_chunk) + len(line) + 1 > max_chars:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk = current_chunk + "\n" + line if current_chunk else line

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


# ─── Consolidation Timer ──────────────────────────────────────


async def consolidation_loop(agent: MemoryAgent, interval_minutes: int = 30):
    """Run consolidation periodically, like sleep cycles."""
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
                result = await agent.consolidate()
                log.info(f"🔄 {result[:100]}")
            else:
                log.info(f"🔄 Skipping consolidation ({count} unconsolidated memories)")
        except Exception as e:
            log.error(f"Consolidation error: {e}")


# ─── Deep Re-Consolidation Loop ───────────────────────────────


async def deep_reconsolidate_loop(agent: MemoryAgent, interval_hours: int = 24):
    """Run deep re-consolidation using a smarter model every 24 hours."""
    log.info(f"🧠 Deep Re-Consolidation: every {interval_hours} hours (model: {SMART_MODEL})")
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
                log.info(f"🧠 Running deep re-consolidation ({total} total memories)...")
                await agent.deep_reconsolidate()
                
                # After factual consolidation, run self-improvement
                log.info("🧬 Triggering self-improvement audit...")
                await agent.self_improve()
            else:
                log.info(f"🧠 Skipping deep re-consolidation ({total} memories, need >= 5)")
        except Exception as e:
            log.error(f"Deep reconsolidation error: {e}")


# ─── Decay Loop ────────────────────────────────────────────────


async def decay_loop(agent: MemoryAgent, interval_minutes: int = 60):
    """Run memory decay periodically to lower importance of old, unconsolidated memories and delete noise."""
    log.info(f"🍂 Decay Loop: every {interval_minutes} minutes")
    while not _shutdown_event.is_set():
        try:
            await asyncio.wait_for(_shutdown_event.wait(), timeout=interval_minutes * 60)
            break
        except asyncio.TimeoutError:
            pass
        try:
            with db_session() as db:
                now_ts = datetime.now(timezone.utc).timestamp()
                cutoff_iso = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
                recent_activity = db.execute(
                    "SELECT COUNT(*) as c FROM memories WHERE created_at > ?", (cutoff_iso,)
                ).fetchone()["c"]

                if recent_activity == 0:
                    log.info("🍂 System idle (no new memories in last 2h). Pausing decay.")
                    continue

                rows = db.execute("SELECT id, importance, created_at FROM memories WHERE consolidated = 0").fetchall()
                culled = 0
                decayed = 0
                for r in rows:
                    created_ts = datetime.fromisoformat(r["created_at"]).replace(tzinfo=timezone.utc).timestamp()
                    age_hours = (now_ts - created_ts) / 3600.0
                    if age_hours > 24.0:
                        new_importance = r["importance"] - 0.05
                        if new_importance < 0.1:
                            db.execute("DELETE FROM memories WHERE id = ?", (r["id"],))
                            culled += 1
                        else:
                            db.execute("UPDATE memories SET importance = ? WHERE id = ?", (new_importance, r["id"]))
                            decayed += 1
                db.commit()
            if culled > 0 or decayed > 0:
                log.info(f"🍂 Decay complete: {decayed} decayed, {culled} culled (importance < 0.1)")
        except Exception as e:
            log.error(f"Decay error: {e}")


# ─── HTTP API ──────────────────────────────────────────────────


def build_http(agent: MemoryAgent, watch_path: str = "./inbox"):
    app = web.Application()

    async def handle_query(request: web.Request):
        q = request.query.get("q", "").strip()
        if not q:
            return web.json_response({"error": "missing ?q= parameter"}, status=400)
        answer = await agent.query(q)
        return web.json_response({"question": q, "answer": answer})

    async def handle_ingest(request: web.Request):
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

    async def handle_consolidate(request: web.Request):
        result = await agent.consolidate()
        return web.json_response({"status": "done", "response": result})

    async def handle_reconsolidate(request: web.Request):
        result = await agent.deep_reconsolidate()
        return web.json_response({"status": "done", "response": result})

    async def handle_improve(request: web.Request):
        result = await agent.self_improve()
        return web.json_response({"status": "done", "response": result})

    async def handle_status(request: web.Request):
        stats = get_memory_stats()
        return web.json_response(stats)

    async def handle_memories(request: web.Request):
        data = read_all_memories()
        return web.json_response(data)

    async def handle_delete(request: web.Request):
        try:
            data = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        memory_id = data.get("memory_id")
        if not memory_id:
            return web.json_response({"error": "missing 'memory_id' field"}, status=400)
        result = delete_memory(int(memory_id))
        return web.json_response(result)

    async def handle_clear(request: web.Request):
        result = clear_all_memories(inbox_path=watch_path)
        return web.json_response(result)

    async def handle_search(request: web.Request):
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


# ─── Main ──────────────────────────────────────────────────────

# Global shutdown coordination
_shutdown_event = asyncio.Event()
_shutting_down = False  # guards against repeated signal log spam


async def main_async(args):
    agent = MemoryAgent()

    log.info("🧠 Agent Memory Layer v2 starting (PydanticAI)")
    log.info(f"   Lite Model: {MODEL}")
    log.info(f"   Smart Model: {SMART_MODEL}")
    log.info(f"   Rate Limit: {RATE_LIMIT} concurrent")
    log.info(f"   Database: {DB_PATH}")
    log.info(f"   Inbox Watch: {args.watch}")
    log.info(f"   Watch Dirs: {WATCH_DIRS or '(none)'}")
    log.info(f"   Consolidate: every {args.consolidate_every}m")
    log.info(f"   API: http://localhost:{args.port}")
    log.info(f"   Vector Search: {'✅' if HAS_SQLITE_VEC else '❌ (install sqlite-vec)'}")
    log.info("")

    # Start background tasks
    tasks = [
        asyncio.create_task(watch_folder(agent, Path(args.watch))),
        asyncio.create_task(consolidation_loop(agent, args.consolidate_every)),
        asyncio.create_task(decay_loop(agent, args.consolidate_every * 2)),
        asyncio.create_task(deep_reconsolidate_loop(agent, 24)),
        asyncio.create_task(index_documents_loop(interval_minutes=args.consolidate_every * 2)),
    ]

    # Start HTTP server
    app = build_http(agent, watch_path=args.watch)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", args.port)
    await site.start()

    log.info(f"✅ Agent running. Drop files in {args.watch}/ or POST to http://localhost:{args.port}/ingest")
    log.info("   Press Ctrl+C to stop.")
    log.info("")

    # Wait for shutdown signal instead of gathering tasks
    try:
        await _shutdown_event.wait()
    finally:
        log.info("🛑 Cancelling background tasks...")
        for t in tasks:
            t.cancel()
        # Wait for all tasks to acknowledge cancellation
        await asyncio.gather(*tasks, return_exceptions=True)
        await runner.cleanup()
        log.info("🧠 Agent stopped.")


def main():
    parser = argparse.ArgumentParser(description="Agent Memory Layer v2 - PydanticAI")
    parser.add_argument("--watch", default="./inbox", help="Folder to watch for new files (default: ./inbox)")
    parser.add_argument("--port", type=int, default=8888, help="HTTP API port (default: 8888)")
    parser.add_argument("--consolidate-every", type=int, default=30, help="Consolidation interval in minutes (default: 30)")
    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Re-create the event in this loop's context
    global _shutdown_event
    _shutdown_event = asyncio.Event()

    def _signal_handler(sig):
        global _shutting_down
        if not _shutting_down:
            _shutting_down = True
            log.info(f"\n👋 Received signal {sig}. Shutting down...")
            _shutdown_event.set()
        else:
            # Second Ctrl+C: force exit immediately
            log.info("\n⚡ Force exit.")
            import os
            os._exit(1)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler, sig)

    try:
        loop.run_until_complete(main_async(args))
    except KeyboardInterrupt:
        # Fallback if signal handler didn't fire (e.g. Windows)
        log.info("\n👋 Keyboard interrupt. Shutting down...")
        _shutdown_event.set()
        # Give tasks a moment to clean up
        loop.run_until_complete(asyncio.sleep(0.5))
    finally:
        # Cancel any remaining tasks
        pending = asyncio.all_tasks(loop)
        for t in pending:
            t.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


if __name__ == "__main__":
    main()
