"""
Librarian Module — Handles document indexing, vector search, and skill management.
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

from config import (
    WATCH_DIRS, IGNORE_DIRS, SKIP_DIRS, CODE_EXTENSIONS, 
    HAS_SQLITE_VEC, SKILLS_DIR, DEBOUNCE_INTERVAL, SCAN_INTERVAL,
    _shutdown_event
)
from database import db_session
from utils import is_binary_file, embed_text, serialize_int8, chunk_text

log = logging.getLogger("memory-agent.librarian")

async def search_documents(query: str, k: int = 5) -> Dict[str, Any]:
    """Search indexed source code and documents by semantic similarity."""
    if not HAS_SQLITE_VEC:
        return {"results": [], "error": "Vector search not available (missing sqlite-vec)"}

    query_embedding = await embed_text(query, shutdown_event=_shutdown_event)
    if not query_embedding:
        return {"results": [], "error": "Failed to generate embedding for query."}

    with db_session() as db:
        rows = db.execute(
            """
            SELECT v.document_id, v.distance, d.path, d.chunk_text, d.chunk_index
            FROM vec_documents v
            JOIN documents d ON d.id = v.document_id
            WHERE v.embedding MATCH vec_int8(?) AND k = ?
            ORDER BY v.distance
            """,
            [serialize_int8(query_embedding), k],
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

def read_document(path: str) -> Dict[str, Any]:
    """Read the full content of a document/file from disk (safe)."""
    file_path = Path(path).resolve()
    
    # Security: check if path is within allowed directories
    # Note: This list should be updated based on actual WATCH_DIRS
    allowed_dirs = [Path("inbox").resolve(), Path(SKILLS_DIR).expanduser().resolve()]
    if WATCH_DIRS:
        allowed_dirs.extend([Path(d.strip()).expanduser().resolve() for d in WATCH_DIRS.split(",") if d.strip()])
    
    # Improved security: use is_relative_to if available (Python 3.9+)
    is_allowed = False
    for d in allowed_dirs:
        try:
            file_path.relative_to(d)
            is_allowed = True
            break
        except ValueError:
            continue

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

def write_skill_file(skill_name: str, content: str) -> Dict[str, Any]:
    """Writes a new skill or rule to the configured SKILLS_DIR."""
    clean_name = "".join(c for c in skill_name if c.isalnum() or c in ("-", "_")).lower()
    target_dir = Path(SKILLS_DIR) / clean_name
    target_dir.mkdir(parents=True, exist_ok=True)
    
    skill_path = target_dir / "SKILL.md"
    skill_path.write_text(content, encoding="utf-8")
    
    log.info(f"💾 Saved skill: {clean_name} to {skill_path}")
    return {"status": "saved", "path": str(skill_path), "skill_name": clean_name}

def _get_latest_mtime(dirs: List[str]) -> float:
    """Find the max modification time across all relevant files."""
    extra_ignores = {d.strip() for d in IGNORE_DIRS.split(",") if d.strip()}
    all_skip = SKIP_DIRS | extra_ignores

    latest_mtime = 0.0
    for dir_path in dirs:
        folder = Path(dir_path).expanduser().resolve()
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

async def _check_semantic_drift(path: str, new_embeddings: List[List[float]], on_drift_detected: Any):
    """
    Check if current active memories linked to this path have drifted semantically.
    """
    from config import DRIFT_THRESHOLD
    import numpy as np
    
    # Calculate centroid of new embeddings
    centroid = np.mean(new_embeddings, axis=0)
    
    # TurboQuant: apply the same rotation to the centroid
    from turboquant import get_turboquant
    tq = get_turboquant(dim=len(centroid))
    # transform() handles normalization and rotation
    rotated_centroid = tq.transform(centroid)

    with db_session() as db:

        # Find memories with an active file_link to this path
        # SQLite JSON path search: connections is a list of objects
        rows = db.execute(
            """
            SELECT m.id, vec_to_json(v.embedding) as vector, m.connections 
            FROM memories m
            JOIN vec_memories v ON m.id = v.memory_id
            WHERE m.connections LIKE ?
            """,
            (f'%"{path}"%',) # Rough filter, refined below
        ).fetchall()

    for r in rows:
        connections = json.loads(r["connections"])
        is_active_link = any(
            c.get("type") == "file_link" and 
            c.get("path") == path and 
            c.get("status", "active") == "active"
            for c in connections
        )
        
        if not is_active_link:
            continue

        memory_id = r["id"]
        memory_vector = json.loads(r["vector"])
        memory_emb = np.array(memory_vector)
        m_norm = np.linalg.norm(memory_emb)
        if m_norm > 1e-9:
            memory_emb = memory_emb / m_norm
            
        # Cosine distance = 1 - cosine similarity
        distance = 1.0 - np.dot(rotated_centroid, memory_emb)

        
        if distance > DRIFT_THRESHOLD:
            log.info(f"🚨 Drift detected for memory #{memory_id} on '{path}' (dist: {distance:.3f} > {DRIFT_THRESHOLD})")
            if asyncio.iscoroutinefunction(on_drift_detected):
                await on_drift_detected(path, memory_id)
            else:
                on_drift_detected(path, memory_id)

async def index_all_dirs(dirs: List[str], on_drift_detected: Any = None):
    """Walk directories and index files."""
    indexed = 0
    skipped = 0

    for dir_path in dirs:
        folder = Path(dir_path).expanduser().resolve()
        if not folder.is_dir():
            continue

        extra_ignores = {d.strip() for d in IGNORE_DIRS.split(",") if d.strip()}
        all_skip = SKIP_DIRS | extra_ignores

        for f in folder.rglob("*"):
            if _shutdown_event.is_set():
                return

            if not f.is_file() or f.suffix.lower() not in CODE_EXTENSIONS:
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

            with db_session() as db:
                existing = db.execute(
                    "SELECT content_hash FROM documents WHERE path = ? AND chunk_index = 0",
                    (str(f),)
                ).fetchone()

                if existing and existing["content_hash"] == content_hash:
                    skipped += 1
                    continue

                db.execute("DELETE FROM documents WHERE path = ?", (str(f),))
                db.commit()

            chunks = chunk_text(text, max_chars=1500)
            log.info(f"📚 Indexing '{f.name}': {len(chunks)} chunks...")
            now = datetime.now(timezone.utc).isoformat()
            
            chunks_embeddings = []
            for i, chunk in enumerate(chunks):
                if _shutdown_event.is_set():
                    break

                embedding = await embed_text(chunk, shutdown_event=_shutdown_event)
                if embedding:
                    chunks_embeddings.append(embedding)
                
                with db_session() as db: 
                    cursor = db.execute(
                        "INSERT INTO documents (path, content_hash, chunk_text, chunk_index, updated_at) VALUES (?, ?, ?, ?, ?)",
                        (str(f), content_hash, chunk, i, now),
                    )
                    doc_id = cursor.lastrowid
                    
                    if embedding:
                        try:
                            # Note: sqlite-vec int8 expects 3072 dims by default in database.py
                            db.execute(
                                "INSERT INTO vec_documents (document_id, embedding) VALUES (?, vec_int8(?))",
                                (doc_id, serialize_int8(embedding)),
                            )
                        except Exception as e:
                            log.error(f"Vec insert error for {f.name}: {e}")
                    db.commit()

            indexed += 1
            
            # Semantic Drift Detection (V3.3)
            if on_drift_detected and chunks_embeddings:
                await _check_semantic_drift(str(f), chunks_embeddings, on_drift_detected)

    if indexed > 0 or skipped > 0:
        log.info(f"📚 Indexing complete: {indexed} files indexed, {skipped} unchanged")

async def librarian_loop(on_drift_detected: Any = None):
    """Periodically index documents."""
    log.info("📚 Librarian loop started.")
    if not WATCH_DIRS:
        log.info("📚 Librarian: WATCH_DIRS is empty. Skipping indexer.")
        return
    if not HAS_SQLITE_VEC:
        log.warning("📚 Librarian: sqlite-vec not found. Skipping vector indexer.")
        return

    dirs = [d.strip() for d in WATCH_DIRS.split(",") if d.strip()]
    
    # Resolve and expand all dirs once at startup
    dirs = [str(Path(d).expanduser().resolve()) for d in dirs]
    
    with db_session() as db:
        row = db.execute("SELECT MAX(updated_at) as last_idx FROM documents").fetchone()
        last_indexed_iso = row["last_idx"] if row and row["last_idx"] else "1970-01-01T00:00:00"
        last_indexed_time = datetime.fromisoformat(last_indexed_iso).replace(tzinfo=timezone.utc).timestamp()

    last_change_time = None
    current_max_mtime = last_indexed_time
    
    while not _shutdown_event.is_set():
        try:
            latest_mtime = _get_latest_mtime(dirs)
            if latest_mtime > current_max_mtime:
                log.info(f"📚 Librarian: detected changes (mtime {latest_mtime} > {current_max_mtime}). Starting debounce timer.")
                last_change_time = time.time()
                current_max_mtime = latest_mtime
            
            if last_change_time:
                elapsed = time.time() - last_change_time
                if elapsed >= DEBOUNCE_INTERVAL:
                    log.info(f"📚 Librarian: debounce window ({DEBOUNCE_INTERVAL}s) closed. Synchronizing vector index...")
                    await index_all_dirs(dirs, on_drift_detected=on_drift_detected)
                    
                    with db_session() as db:
                        row = db.execute("SELECT MAX(updated_at) as last_idx FROM documents").fetchone()
                        if row and row["last_idx"]:
                            last_indexed_time = datetime.fromisoformat(row["last_idx"]).replace(tzinfo=timezone.utc).timestamp()
                    
                    last_change_time = None
                    current_max_mtime = last_indexed_time
                else:
                    if int(elapsed) % 15 == 0: # Log every 15s of debounce
                        log.debug(f"📚 Librarian: debouncing... ({int(elapsed)}/{DEBOUNCE_INTERVAL}s)")
        except Exception as e:
            log.error(f"Indexing error: {e}")

        try:
            await asyncio.wait_for(_shutdown_event.wait(), timeout=SCAN_INTERVAL)
            break
        except asyncio.TimeoutError:
            pass
