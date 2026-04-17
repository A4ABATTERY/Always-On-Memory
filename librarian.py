"""
Librarian Module — Handles document indexing, vector search, and skill management.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

from config import (
    WATCH_DIRS, IGNORE_DIRS, SKIP_DIRS, CODE_EXTENSIONS,
    HAS_SQLITE_VEC, SKILLS_DIR, DEBOUNCE_INTERVAL, SCAN_INTERVAL,
    get_shutdown_event, INBOX_DIR, PROMOTION_THRESHOLD
)
from database import db_session
from utils import is_binary_file, embed_text, embed_texts_batch, serialize_int8, chunk_text, chunk_code_structural
from lexical_parser import extract_symbols

log = logging.getLogger("memory-agent.librarian")

async def search_documents(query: str, k: int = 5) -> Dict[str, Any]:
    """Search indexed source code and documents by semantic similarity."""
    if not HAS_SQLITE_VEC:
        return {"results": [], "error": "Vector search not available (missing sqlite-vec)"}

    query_embedding = await embed_text(query, task_type="retrieval", shutdown_event=get_shutdown_event())
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

def search_symbols(query: str, k: int = 10) -> Dict[str, Any]:
    """Search the Lexical Symbol Index for exact or prefix matches on identifier names.

    Use this for identifier lookups: function names, class names, constant names.
    Do NOT use for semantic/conceptual queries — use search_documents for those.

    Returns file_path, symbol_type, and line_no so the agent can cite the exact location.
    """
    with db_session() as db:
        # 1. Exact match (case-insensitive)
        rows = db.execute(
            "SELECT file_path, symbol_name, symbol_type, line_no, signature "
            "FROM symbols WHERE symbol_name = ? COLLATE NOCASE LIMIT ?",
            (query, k),
        ).fetchall()

        match_type = "exact"
        if not rows:
            # 2. Prefix match fallback
            rows = db.execute(
                "SELECT file_path, symbol_name, symbol_type, line_no, signature "
                "FROM symbols WHERE symbol_name LIKE ? COLLATE NOCASE LIMIT ?",
                (query + "%", k),
            ).fetchall()
            match_type = "prefix"

    results = [
        {
            "file_path": r["file_path"],
            "symbol_name": r["symbol_name"],
            "symbol_type": r["symbol_type"],
            "line_no": r["line_no"],
            "signature": r["signature"],
        }
        for r in rows
    ]
    return {"results": results, "count": len(results), "match_type": match_type}


def read_document(path: str) -> Dict[str, Any]:
    """Read the full content of a document/file from disk (safe)."""
    file_path = Path(path).resolve()
    
    # Security: check if path is within allowed directories
    # Note: This list should be updated based on actual WATCH_DIRS
    allowed_dirs = [Path(INBOX_DIR).resolve(), Path(SKILLS_DIR).expanduser().resolve()]
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
    """Find the max modification time across all relevant files using os.scandir for speed."""
    extra_ignores = {d.strip(' \t\n\r"\'') for d in IGNORE_DIRS.split(",") if d.strip()}
    all_skip = SKIP_DIRS | extra_ignores
    abs_extra = {os.path.normcase(os.path.abspath(e)) for e in extra_ignores if os.path.isabs(e)}

    latest_mtime = 0.0

    def _walk_recursive(current_path: str) -> float:
        current_max = 0.0
        try:
            # os.scandir is much faster than Path.rglob as it doesn't walk ignored branches
            with os.scandir(current_path) as it:
                for entry in it:
                    # Early skip: dotfiles or ignored names (node_modules, .git, etc.)
                    if entry.name.startswith(".") or entry.name in all_skip:
                        continue
                    
                    entry_abs = os.path.normcase(entry.path)
                    # Absolute ignore check
                    if any(entry_abs == ignore or entry_abs.startswith(ignore + os.sep) for ignore in abs_extra):
                        continue
                    
                    if entry.is_dir(follow_symlinks=False):
                        current_max = max(current_max, _walk_recursive(entry.path))
                    elif entry.is_file(follow_symlinks=False):
                        _, ext = os.path.splitext(entry.name)
                        if ext.lower() in CODE_EXTENSIONS:
                            # Verification: stat() is called on the DirEntry (often cached by OS)
                            mtime = entry.stat().st_mtime
                            if mtime > current_max:
                                current_max = mtime
        except (OSError, PermissionError):
            pass
        return current_max

    for dir_path in dirs:
        root = os.path.normpath(os.path.abspath(os.path.expanduser(dir_path)))
        if os.path.isdir(root):
            latest_mtime = max(latest_mtime, _walk_recursive(root))
            
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
            (f'%{os.path.basename(path)}%',) # Pre-filter by filename to avoid full table scan
        ).fetchall()

    for r in rows:
        connections = json.loads(r["connections"])
        is_active_link = any(
            c.get("type") == "file_link" and 
            os.path.normcase(os.path.abspath(c.get("path", ""))) == os.path.normcase(os.path.abspath(path)) and 
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

def _has_memory_tag(text: str) -> bool:
    """Return True if the file contains a '# @memory' promotion tag in a comment line.

    Only matches lines where '#' is the first non-whitespace character, preventing
    false positives from strings or docstrings like '"Use # @memory to promote."'.
    """
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#") and "@memory" in stripped:
            return True
    return False


async def _max_drift_score(path: str, new_embeddings: List[List[float]]) -> float:
    """Return the maximum cosine drift distance for memories linked to this file."""
    if not HAS_SQLITE_VEC or not new_embeddings:
        return 0.0
    import numpy as np
    from turboquant import get_turboquant

    centroid = np.mean(new_embeddings, axis=0)
    tq = get_turboquant(dim=len(centroid))
    rotated = tq.transform(centroid)

    max_dist = 0.0
    with db_session() as db:
        rows = db.execute(
            "SELECT vec_to_json(v.embedding) as vector FROM memories m "
            "JOIN vec_memories v ON m.id = v.memory_id WHERE m.connections LIKE ?",
            (f'%{os.path.basename(path)}%',),
        ).fetchall()
    for r in rows:
        mem_emb = np.array(json.loads(r["vector"]))
        norm = np.linalg.norm(mem_emb)
        if norm > 1e-9:
            mem_emb = mem_emb / norm
        dist = 1.0 - np.dot(rotated, mem_emb)
        if dist > max_dist:
            max_dist = dist
    return max_dist


async def index_all_dirs(dirs: List[str], on_drift_detected: Any = None, on_promotion_triggered: Any = None):
    """Walk directories and index files."""
    indexed = 0
    skipped = 0

    for dir_path in dirs:
        folder = Path(dir_path).expanduser().resolve()
        if not folder.is_dir():
            continue

        extra_ignores = {d.strip(' \t\n\r"\'') for d in IGNORE_DIRS.split(",") if d.strip()}
        all_skip = SKIP_DIRS | extra_ignores
        abs_extra = {os.path.normcase(os.path.abspath(e)) for e in extra_ignores if os.path.isabs(e)}

        for f in folder.rglob("*"):
            if get_shutdown_event().is_set():
                return

            if not f.is_file() or f.suffix.lower() not in CODE_EXTENSIONS:
                continue

            rel_parts = f.relative_to(folder).parts
            if any(p.startswith(".") or p in all_skip for p in rel_parts):
                continue

            f_abs = os.path.normcase(os.path.abspath(str(f)))
            if any(f_abs.startswith(ignore + os.sep) or f_abs == ignore for ignore in abs_extra):
                continue
            
            if is_binary_file(f):
                continue

            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue

            content_hash = hashlib.md5(text.encode()).hexdigest()
            f_abs = os.path.normcase(os.path.abspath(str(f)))

            with db_session() as db:
                existing = db.execute(
                    "SELECT content_hash FROM documents WHERE path = ? AND chunk_index = 0",
                    (f_abs,)
                ).fetchone()

                if existing and existing["content_hash"] == content_hash:
                    skipped += 1
                    continue

                # Delete all previous chunks for this file before re-indexing
                db.execute("DELETE FROM documents WHERE lower(path) = ?", (f_abs.lower(),))
                db.commit()

            chunks = chunk_code_structural(text, f.suffix)
            log.info(f"📚 Indexing '{f.name}': {len(chunks)} structural chunks (1 batch embed call)...")
            now = datetime.now(timezone.utc).isoformat()

            if get_shutdown_event().is_set():
                break

            # Batch-embed all chunks in a single API call instead of N serial calls.
            # Now passing filename as title to support Gemini V2 document prefixing.
            all_embeddings = await embed_texts_batch(
                chunks, 
                task_type="document", 
                titles=[f.name] * len(chunks),
                shutdown_event=get_shutdown_event()
            )

            chunks_embeddings = []
            for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
                if get_shutdown_event().is_set():
                    break

                if embedding:
                    chunks_embeddings.append(embedding)

                with db_session() as db:
                    db.execute(
                        "INSERT INTO documents (path, content_hash, chunk_text, chunk_index, updated_at) VALUES (?, ?, ?, ?, ?)",
                        (f_abs, content_hash, chunk, i, now),
                    )
                    doc_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

                    if embedding:
                        try:
                            db.execute(
                                "INSERT INTO vec_documents (document_id, embedding) VALUES (?, vec_int8(?))",
                                (doc_id, serialize_int8(embedding)),
                            )
                        except Exception as e:
                            log.error(f"Vec insert error for {f.name}: {e}")
                    db.commit()

            indexed += 1

            # Lexical Symbol Index — upsert named identifiers for this file.
            symbols = extract_symbols(text, f.suffix)
            if symbols:
                now_sym = datetime.now(timezone.utc).isoformat()
                with db_session() as db:
                    db.execute("DELETE FROM symbols WHERE file_path = ?", (f_abs,))
                    db.executemany(
                        "INSERT INTO symbols (file_path, symbol_name, symbol_type, line_no, signature, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        [
                            (f_abs, s["name"], s["type"], s["line_no"], s["signature"], now_sym)
                            for s in symbols
                        ],
                    )
                    db.commit()
                log.debug(f"📖 LSI: indexed {len(symbols)} symbols from '{f.name}'")

            # Semantic Drift Detection (V3.3)
            if on_drift_detected and chunks_embeddings:
                await _check_semantic_drift(f_abs, chunks_embeddings, on_drift_detected)

            # Promotion Logic — full semantic ingest for files tagged # @memory
            # or with high drift score. Hash-gated to prevent re-promotion on
            # every indexing cycle when file content hasn't changed.
            if on_promotion_triggered:
                tagged = _has_memory_tag(text)
                drift_score = 0.0
                if not tagged and chunks_embeddings:
                    drift_score = await _max_drift_score(f_abs, chunks_embeddings)
                should_promote = tagged or drift_score > PROMOTION_THRESHOLD

                if should_promote:
                    with db_session() as db:
                        row = db.execute(
                            "SELECT promoted_hash FROM documents WHERE path = ? AND chunk_index = 0",
                            (f_abs,),
                        ).fetchone()
                    already_promoted = row and row["promoted_hash"] == content_hash

                    if not already_promoted:
                        reason = "# @memory tag" if tagged else f"drift={drift_score:.3f}"
                        log.info(f"📢 Promoting '{f.name}' to Ingest Agent ({reason})")
                        if asyncio.iscoroutinefunction(on_promotion_triggered):
                            await on_promotion_triggered(f_abs, text)
                        else:
                            on_promotion_triggered(f_abs, text)
                        with db_session() as db:
                            db.execute(
                                "UPDATE documents SET promoted_hash = ? WHERE path = ? AND chunk_index = 0",
                                (content_hash, f_abs),
                            )
                            db.commit()

    if indexed > 0 or skipped > 0:
        log.info(f"📚 Indexing complete: {indexed} files indexed, {skipped} unchanged")

async def librarian_loop(on_drift_detected: Any = None, on_promotion_triggered: Any = None):
    """Periodically index documents."""
    log.info("📚 Librarian loop started.")
    if not WATCH_DIRS:
        log.info("📚 Librarian: WATCH_DIRS is empty. Skipping indexer.")
        return
    if not HAS_SQLITE_VEC:
        log.warning("📚 Librarian: sqlite-vec not found. Skipping vector indexer.")
        return

    dirs = [d.strip(' \t\n\r"\'') for d in WATCH_DIRS.split(",") if d.strip()]
    
    # Resolve and expand all dirs once at startup
    dirs = [str(Path(d).expanduser().resolve()) for d in dirs]
    
    with db_session() as db:
        row = db.execute("SELECT MAX(updated_at) as last_idx FROM documents").fetchone()
        last_indexed_iso = row["last_idx"] if row and row["last_idx"] else "1970-01-01T00:00:00"
        last_indexed_time = datetime.fromisoformat(last_indexed_iso).replace(tzinfo=timezone.utc).timestamp()

    last_change_time = None
    current_max_mtime = last_indexed_time
    
    while not get_shutdown_event().is_set():
        try:
            # Shift FS-heavy mtime check to a background thread to keep loop responsive
            latest_mtime = await asyncio.to_thread(_get_latest_mtime, dirs)
            if latest_mtime > current_max_mtime:
                log.info(f"📚 Librarian: detected changes (mtime {latest_mtime} > {current_max_mtime}). Starting debounce timer.")
                last_change_time = time.time()
                current_max_mtime = latest_mtime
            
            if last_change_time:
                elapsed = time.time() - last_change_time
                if elapsed >= DEBOUNCE_INTERVAL:
                    log.info(f"📚 Librarian: debounce window ({DEBOUNCE_INTERVAL}s) closed. Synchronizing vector index...")
                    await index_all_dirs(dirs, on_drift_detected=on_drift_detected, on_promotion_triggered=on_promotion_triggered)
                    
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
            await asyncio.wait_for(get_shutdown_event().wait(), timeout=SCAN_INTERVAL)
            break
        except asyncio.TimeoutError:
            pass
