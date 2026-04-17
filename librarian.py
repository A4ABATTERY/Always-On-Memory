"""
Librarian Module — Handles document indexing, vector search, and skill management.
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

from config import (
    WATCH_DIRS, IGNORE_DIRS, SKIP_DIRS, CODE_EXTENSIONS,
    HAS_SQLITE_VEC, SKILLS_DIR, DEBOUNCE_INTERVAL, SCAN_INTERVAL,
    get_shutdown_event, INBOX_DIR, PROMOTION_THRESHOLD, VERIFY_INTERVAL_HOURS
)
from database import db_session
from utils import is_binary_file, embed_text, embed_texts_batch, serialize_int8, chunk_text, chunk_code_structural
from lexical_parser import extract_symbols

log = logging.getLogger("memory-agent.librarian")


def get_resolved_watch_dirs() -> List[str]:
    """Resolve WATCH_DIRS env var into a list of normalized absolute paths.

    Single source of truth used by librarian_loop, verification_loop, and
    the REST /verify endpoint — avoids duplicating path resolution logic.
    """
    dirs = [d.strip(' \t\n\r"\'') for d in WATCH_DIRS.split(",") if d.strip()]
    return [str(Path(d).expanduser().resolve()) for d in dirs]


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

                # Delete all previous chunks for this file before re-indexing.
                # vec_documents must be cleaned first — vec0 virtual tables do not
                # cascade FK deletes, so old rows with stale document_ids would
                # otherwise accumulate and pollute semantic search results.
                db.execute(
                    "DELETE FROM vec_documents WHERE document_id IN "
                    "(SELECT id FROM documents WHERE path = ?)",
                    (f_abs,)
                )
                db.execute("DELETE FROM documents WHERE path = ?", (f_abs,))
                db.commit()

            chunks = chunk_code_structural(text, f.suffix)
            n_calls = math.ceil(len(chunks) / 100)
            call_label = f"{n_calls} batch embed call{'s' if n_calls > 1 else ''}"
            log.info(f"📚 Indexing '{f.name}': {len(chunks)} structural chunks ({call_label})...")
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

    dirs = get_resolved_watch_dirs()
    
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


# ─── Verification Pass ─────────────────────────────────────────


async def index_single_file(
    path: str,
    on_drift_detected: Any = None,
    on_promotion_triggered: Any = None,
) -> bool:
    """Re-index a single file by absolute path, always overwriting existing data.

    This is the repair path used by the verification pass. Unlike index_all_dirs,
    it never checks the content hash — it always re-indexes regardless of what
    is already stored, since the goal is to fix incomplete or missing chunks.

    Cleanup order:
      1. vec_documents (by document_id) — vec0 tables don't cascade FK deletes
      2. documents — remove stale chunk rows
      3. symbols — remove stale LSI entries

    promoted_hash is preserved when the file content is unchanged (same hash),
    preventing a re-promotion of already-promoted files on the next librarian pass.
    """
    f = Path(path)
    if not f.is_file():
        raise FileNotFoundError(f"index_single_file: not a file: {path}")

    try:
        text = f.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        raise IOError(f"index_single_file: cannot read {path}: {e}") from e

    content_hash = hashlib.md5(text.encode()).hexdigest()
    now = datetime.now(timezone.utc).isoformat()

    # Save promoted_hash before wiping rows so we can restore it if content unchanged
    saved_promoted_hash = None
    with db_session() as db:
        row = db.execute(
            "SELECT content_hash, promoted_hash FROM documents WHERE path = ? AND chunk_index = 0",
            (path,)
        ).fetchone()
        if row:
            saved_promoted_hash = row["promoted_hash"] if row["content_hash"] == content_hash else None

        # Delete in correct order: vectors first, then chunk rows, then symbols
        db.execute(
            "DELETE FROM vec_documents WHERE document_id IN "
            "(SELECT id FROM documents WHERE path = ?)",
            (path,)
        )
        db.execute("DELETE FROM documents WHERE path = ?", (path,))
        db.execute("DELETE FROM symbols WHERE file_path = ?", (path,))
        db.commit()

    chunks = chunk_code_structural(text, f.suffix)
    n_calls = math.ceil(len(chunks) / 100)
    call_label = f"{n_calls} batch embed call{'s' if n_calls > 1 else ''}"
    log.info(f"📋 Verifier re-indexing '{f.name}': {len(chunks)} chunks ({call_label})...")

    if get_shutdown_event().is_set():
        return False

    all_embeddings = await embed_texts_batch(
        chunks,
        task_type="document",
        titles=[f.name] * len(chunks),
        shutdown_event=get_shutdown_event(),
    )

    chunks_embeddings = []
    for i, (chunk, embedding) in enumerate(zip(chunks, all_embeddings)):
        if get_shutdown_event().is_set():
            break

        if embedding:
            chunks_embeddings.append(embedding)

        with db_session() as db:
            db.execute(
                "INSERT INTO documents (path, content_hash, chunk_text, chunk_index, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (path, content_hash, chunk, i, now),
            )
            doc_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]

            if embedding:
                try:
                    db.execute(
                        "INSERT INTO vec_documents (document_id, embedding) VALUES (?, vec_int8(?))",
                        (doc_id, serialize_int8(embedding)),
                    )
                except Exception as e:
                    log.error(f"Vec insert error for {f.name} chunk {i}: {e}")
            db.commit()

    # Restore promoted_hash if content unchanged — prevents spurious re-promotion
    if saved_promoted_hash is not None:
        with db_session() as db:
            db.execute(
                "UPDATE documents SET promoted_hash = ? WHERE path = ? AND chunk_index = 0",
                (saved_promoted_hash, path),
            )
            db.commit()

    # LSI symbol index
    symbols = extract_symbols(text, f.suffix)
    if symbols:
        now_sym = datetime.now(timezone.utc).isoformat()
        with db_session() as db:
            db.executemany(
                "INSERT INTO symbols (file_path, symbol_name, symbol_type, line_no, signature, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [(path, s["name"], s["type"], s["line_no"], s["signature"], now_sym) for s in symbols],
            )
            db.commit()

    # Semantic drift detection
    if on_drift_detected and chunks_embeddings:
        await _check_semantic_drift(path, chunks_embeddings, on_drift_detected)

    # Promotion check (two-argument call matching index_all_dirs pattern)
    if on_promotion_triggered and saved_promoted_hash is None:
        tagged = _has_memory_tag(text)
        drift_score = 0.0
        if not tagged and chunks_embeddings:
            drift_score = await _max_drift_score(path, chunks_embeddings)
        if tagged or drift_score > PROMOTION_THRESHOLD:
            log.info(f"📢 Verifier: promoting '{f.name}' to Ingest Agent")
            if asyncio.iscoroutinefunction(on_promotion_triggered):
                await on_promotion_triggered(path, text)
            else:
                on_promotion_triggered(path, text)
            with db_session() as db:
                db.execute(
                    "UPDATE documents SET promoted_hash = ? WHERE path = ? AND chunk_index = 0",
                    (content_hash, path),
                )
                db.commit()

    log.info(f"✅ Verifier: '{f.name}' successfully re-indexed ({len(chunks)} chunks)")
    return True


async def verify_watch_dirs(
    retry_state: Dict[str, int],
    on_drift_detected: Any = None,
    on_promotion_triggered: Any = None,
) -> Dict[str, Any]:
    """Walk WATCH_DIRS and detect files that are missing or incompletely indexed.

    Check A: file has no row in documents (entirely absent)
    Check B: file has fewer stored chunks than chunk_code_structural produces

    Files failing either check are retried via index_single_file. The retry
    counter (retry_state) is incremented only on failure and deleted on success.
    Files with retry_state[path] >= 3 are permanently skipped with a warning.
    """
    if not HAS_SQLITE_VEC:
        return {"missing": [], "incomplete": [], "retried_ok": [], "retried_failed": [], "permanent_failures": []}

    dirs = get_resolved_watch_dirs()
    if not dirs:
        return {"missing": [], "incomplete": [], "retried_ok": [], "retried_failed": [], "permanent_failures": []}

    missing: List[str] = []
    incomplete: List[str] = []
    retried_ok: List[str] = []
    retried_failed: List[str] = []
    permanent_failures: List[str] = []

    extra_ignores = {d.strip(' \t\n\r"\'') for d in IGNORE_DIRS.split(",") if d.strip()}
    all_skip = SKIP_DIRS | extra_ignores
    abs_extra = {os.path.normcase(os.path.abspath(e)) for e in extra_ignores if os.path.isabs(e)}

    for dir_path in dirs:
        folder = Path(dir_path)
        if not folder.is_dir():
            continue

        for f in folder.rglob("*"):
            if get_shutdown_event().is_set():
                return {"missing": missing, "incomplete": incomplete,
                        "retried_ok": retried_ok, "retried_failed": retried_failed,
                        "permanent_failures": permanent_failures}

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

            # Check A: entirely absent?
            needs_retry = False
            try:
                with db_session() as db:
                    stored_count = db.execute(
                        "SELECT COUNT(*) as c FROM documents WHERE path = ?", (f_abs,)
                    ).fetchone()["c"]

                if stored_count == 0:
                    missing.append(f_abs)
                    needs_retry = True
                else:
                    # Check B: incomplete chunks?
                    try:
                        text = f.read_text(encoding="utf-8", errors="replace")
                        expected_count = len(chunk_code_structural(text, f.suffix))
                        if stored_count < expected_count:
                            log.debug(
                                f"📋 Verifier: '{f.name}' incomplete "
                                f"({stored_count}/{expected_count} chunks)"
                            )
                            incomplete.append(f_abs)
                            needs_retry = True
                    except Exception as e:
                        log.error(f"📋 Verifier: error reading '{f.name}' for chunk check: {e}")

            except Exception as e:
                log.error(f"📋 Verifier: DB error checking '{f.name}': {e}")
                continue

            if not needs_retry:
                continue

            # Retry logic
            attempt = retry_state.get(f_abs, 0)
            if attempt >= 3:
                log.warning(
                    f"❌ Verifier: '{f.name}' permanently failed after 3 retries — "
                    "manual intervention required"
                )
                permanent_failures.append(f_abs)
                continue

            log.warning(
                f"⚠️  Verifier: '{f.name}' {'missing' if f_abs in missing else 'incomplete'} "
                f"— retry {attempt + 1}/3"
            )
            try:
                await index_single_file(f_abs, on_drift_detected, on_promotion_triggered)
                if f_abs in retry_state:
                    del retry_state[f_abs]
                retried_ok.append(f_abs)
            except Exception as e:
                retry_state[f_abs] = attempt + 1
                log.error(f"⚠️  Verifier: retry {attempt + 1}/3 failed for '{f.name}': {e}")
                retried_failed.append(f_abs)

    total_issues = len(missing) + len(incomplete)
    log.info(
        f"📋 Verifier: scan complete — {len(missing)} missing, {len(incomplete)} incomplete, "
        f"{len(permanent_failures)} permanent failures"
    )
    return {
        "missing": missing,
        "incomplete": incomplete,
        "retried_ok": retried_ok,
        "retried_failed": retried_failed,
        "permanent_failures": permanent_failures,
    }


async def verification_loop(
    retry_state: Dict[str, int],
    on_drift_detected: Any = None,
    on_promotion_triggered: Any = None,
) -> None:
    """Periodically verify that all files in WATCH_DIRS are fully indexed.

    Runs verify_watch_dirs every VERIFY_INTERVAL_HOURS hours. Waits 300 seconds
    at startup to avoid racing with librarian_loop's initial debounce pass.
    """
    if not HAS_SQLITE_VEC:
        log.warning("📋 Verifier: sqlite-vec not found. Skipping verification loop.")
        return
    if not WATCH_DIRS:
        log.info("📋 Verifier: WATCH_DIRS is empty. Skipping verification loop.")
        return

    log.info(
        f"📋 Verifier loop started (interval: {VERIFY_INTERVAL_HOURS}h, "
        "startup delay: 300s)."
    )

    # Startup delay — let librarian_loop complete its initial pass first
    try:
        await asyncio.wait_for(get_shutdown_event().wait(), timeout=300)
        return  # shutdown during delay
    except asyncio.TimeoutError:
        pass

    while not get_shutdown_event().is_set():
        try:
            await verify_watch_dirs(retry_state, on_drift_detected, on_promotion_triggered)
        except Exception as e:
            log.error(f"📋 Verifier loop error: {e}")

        try:
            await asyncio.wait_for(
                get_shutdown_event().wait(),
                timeout=VERIFY_INTERVAL_HOURS * 3600,
            )
            break
        except asyncio.TimeoutError:
            pass
