"""
Memory Store Module — Encapsulates database operations for memories and consolidations.
"""

import json
import logging
import os
from uuid import uuid4
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import numpy as np
from database import db_session, HAS_SQLITE_VEC
from models import MemCube
from utils import serialize_int8, embed_text

log = logging.getLogger("memory-agent.store")

async def store_memory(
    raw_text: str,
    summary: str,
    entities: List[str],
    topics: List[str],
    importance_score: float,
    sector: str = "semantic",
    valid_to: Optional[str] = None,
    source: str = "",
    origin_platform: str = "aom-local",
    metadata: Optional[Dict[str, Any]] = None,
    connections: Optional[List[Dict[str, Any]]] = None,
    embedding: Optional[List[float]] = None,
    cube_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Store a MemCube in the database."""
    if embedding is None:
        embedding = await embed_text(raw_text, task_type="document", title=summary) or []

    with db_session() as db:
        now = datetime.now(timezone.utc).isoformat()
        cube = MemCube(
            cube_id=cube_id or str(uuid4()),
            raw_text=raw_text,
            summary=summary,
            entities=entities,
            topics=topics,
            connections=connections or [],
            importance_score=importance_score,
            sector=sector,
            valid_to=valid_to,
            source=source,
            origin_platform=origin_platform,
            metadata=metadata or {},
            created_at=now,
        )
        
        cursor = db.execute(
            """INSERT INTO memories (
                cube_id, sector, source, origin_platform, raw_text, summary, 
                entities, topics, connections, metadata, importance_score, 
                created_at, valid_to
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                cube.cube_id, cube.sector, cube.source, cube.origin_platform, 
                cube.raw_text, cube.summary, json.dumps(cube.entities), 
                json.dumps(cube.topics), json.dumps(cube.connections), 
                json.dumps(cube.metadata), cube.importance_score, 
                cube.created_at, cube.valid_to
            ),
        )
        mid = cursor.lastrowid
        
        if embedding:
            try:
                db.execute(
                    "INSERT INTO vec_memories (memory_id, embedding) VALUES (?, vec_int8(?))",
                    (mid, serialize_int8(embedding)),
                )
            except Exception as e:
                log.error(f"Vec insert error for memory #{mid}: {e}")
        
        db.commit()
    
    log.info(f"📥 Stored MemCube #{mid} [{cube.cube_id}]: {summary[:60]}...")
    return {"memory_id": mid, "cube_id": cube.cube_id, "status": "stored"}

def read_all_memories(limit: int = 200) -> Dict[str, Any]:
    """Read and rank stored memories based on composite score."""
    with db_session() as db:
        now_iso = datetime.now(timezone.utc).isoformat()
        now_ts = datetime.now(timezone.utc).timestamp()

        rows = db.execute(
            "SELECT * FROM memories WHERE valid_to IS NULL OR valid_to > ? ORDER BY created_at DESC LIMIT ?",
            (now_iso, limit)
        ).fetchall()

    memories = []
    for r in rows:
        created_dt = datetime.fromisoformat(r["created_at"])
        if created_dt.tzinfo is None:
            created_dt = created_dt.replace(tzinfo=timezone.utc)
        
        created_ts = created_dt.timestamp()
        age_hours = (now_ts - created_ts) / 3600.0
        # Decay factor: importance decreases by half every 24 hours of age
        score = r["importance_score"] * (1.0 / (1.0 + (age_hours / 24.0)))

        memories.append({
            "id": r["id"], "cube_id": r["cube_id"], "source": r["source"], 
            "origin_platform": r["origin_platform"], "summary": r["summary"],
            "entities": json.loads(r["entities"]), "topics": json.loads(r["topics"]),
            "importance_score": r["importance_score"], 
            "connections": json.loads(r["connections"]),
            "metadata": json.loads(r["metadata"]),
            "access_count": r["access_count"], "last_accessed": r["last_accessed"],
            "created_at": r["created_at"], "consolidated": bool(r["consolidated"]),
            "sector": r["sector"], "valid_to": r["valid_to"],
            "composite_score": score,
            "recall_reason": [
                "High Importance" if r["importance_score"] > 0.7 else "Standard",
                f"Age: {age_hours:.1f}h",
                f"Score: {score:.2f}"
            ]
        })

    memories.sort(key=lambda x: x["composite_score"], reverse=True)
    memories = memories[:100] # Increase limit for results
    return {"memories": memories, "count": len(memories)}

def read_unconsolidated_memories(limit: int = 30) -> Dict[str, Any]:
    """Read memories that haven't been consolidated yet.

    Insight Cubes produced by synthesis passes are excluded so they don't
    re-enter the 30-minute consolidation loop and cause compounding API calls.
    Deep reconsolidation uses read_all_memories() and still processes them.
    """
    _SYNTHESIS_SOURCES = (
        "adversarial-consolidation",
        "deep-reconsolidation",
        "autodream-consolidation",
    )
    placeholders = ",".join("?" * len(_SYNTHESIS_SOURCES))
    with db_session() as db:
        now_iso = datetime.now(timezone.utc).isoformat()
        rows = db.execute(
            f"SELECT * FROM memories WHERE consolidated = 0 "
            f"AND (valid_to IS NULL OR valid_to > ?) "
            f"AND (source IS NULL OR source NOT IN ({placeholders})) "
            f"ORDER BY created_at DESC LIMIT ?",
            (*_SYNTHESIS_SOURCES, now_iso, limit),
        ).fetchall()
    memories = []
    for r in rows:
        memories.append({
            "id": r["id"], "cube_id": r["cube_id"], "summary": r["summary"],
            "entities": json.loads(r["entities"]), "topics": json.loads(r["topics"]),
            "importance_score": r["importance_score"], "created_at": r["created_at"],
        })
    return {"memories": memories, "count": len(memories)}

def read_unconsolidated_with_embeddings(limit: int = 100) -> List[Dict[str, Any]]:
    """Fetch unconsolidated memories along with their embeddings."""
    if not HAS_SQLITE_VEC:
        log.warning("sqlite-vec not available, cannot fetch embeddings.")
        return []

    with db_session() as db:
        # Join memories with vec_memories to get the embeddings
        # We use vec_to_json to convert the int8 quantized vector back to a JSON array of floats
        now_iso = datetime.now(timezone.utc).isoformat()
        rows = db.execute(
            """
            SELECT m.*, vec_to_json(v.embedding) as vector 
            FROM memories m
            JOIN vec_memories v ON m.id = v.memory_id
            WHERE m.consolidated = 0 AND (m.valid_to IS NULL OR m.valid_to > ?)
            ORDER BY m.created_at DESC
            LIMIT ?
            """,
            (now_iso, limit)
        ).fetchall()
    
    results = []
    for r in rows:
        results.append({
            "id": r["id"],
            "cube_id": r["cube_id"],
            "summary": r["summary"],
            "topics": json.loads(r["topics"]),
            "embedding": json.loads(r["vector"]) if isinstance(r["vector"], str) else r["vector"],
            "created_at": r["created_at"]
        })
    return results

def cluster_memories_by_embedding(memories: List[Dict[str, Any]], threshold: float = 0.75) -> Dict[str, List[Dict[str, Any]]]:
    """
    Greedy Similarity Clustering (Leader Algorithm).
    Groups memories into clusters based on cosine similarity.
    """
    if not memories:
        return {}

    clusters: List[Dict[str, Any]] = [] # List of {centroid: np.array, members: list}

    for m in memories:
        emb = np.array(m["embedding"])
        # Normalize for cosine similarity (dot product of unit vectors)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        
        best_sim = -1.0
        best_cluster_idx = -1
        
        for i, cluster in enumerate(clusters):
            # Cosine similarity = dot product since embeddings are normalized
            sim = np.dot(emb, cluster["centroid"])
            if sim > best_sim:
                best_sim = sim
                best_cluster_idx = i
        
        if best_sim >= threshold:
            clusters[best_cluster_idx]["members"].append(m)
            # Optional: update centroid (running average)
            # clusters[best_cluster_idx]["centroid"] = (clusters[best_cluster_idx]["centroid"] + emb) / 2
        else:
            clusters.append({
                "centroid": emb,
                "members": [m]
            })

    # Convert back to a dictionary of clusters for compatibility
    res = {}
    for idx, cluster in enumerate(clusters):
        # Name clusters by the first memory summary or index
        first_mem = cluster["members"][0]
        cluster_name = f"cluster_{idx}_{first_mem['summary'][:20].strip()}"
        res[cluster_name] = cluster["members"]

    log.info(f"🧩 Clustering identified {len(res)} semantic groups from {len(memories)} memories.")
    return res

def read_memory_partition(sector: str, limit: int = 100) -> Dict[str, Any]:
    """Fetches memories from a specific sector."""
    with db_session() as db:
        rows = db.execute(
            "SELECT * FROM memories WHERE sector = ? AND (valid_to IS NULL OR valid_to > datetime('now')) ORDER BY created_at DESC LIMIT ?",
            (sector, limit)
        ).fetchall()
    memories = []
    for r in rows:
        memories.append({
            "id": r["id"], "cube_id": r["cube_id"], "summary": r["summary"], 
            "raw_text": r["raw_text"], "importance_score": r["importance_score"], 
            "created_at": r["created_at"],
        })
    return {"sector": sector, "memories": memories, "count": len(memories)}

def store_consolidation(
    source_ids: List[int],
    summary: str,
    insight: str,
    connections: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Store a consolidation result and mark source memories as consolidated."""
    with db_session() as db:
        now = datetime.now(timezone.utc).isoformat()
        db.execute(
            "INSERT INTO consolidations (source_ids, summary, insight, created_at) VALUES (?, ?, ?, ?)",
            (json.dumps(source_ids), summary, insight, now),
        )
        for conn in connections:
            if not isinstance(conn, dict):
                log.warning(f"Skipping malformed connection (expected dict, got {type(conn)}): {conn}")
                continue
            
            conn_type = conn.get("type", "memory_link")
            rel = str(conn.get("relationship", ""))

            if conn_type == "memory_link":
                from_id, to_id = conn.get("from_id"), conn.get("to_id")
                if from_id and to_id:
                    for mid in [from_id, to_id]:
                        row = db.execute("SELECT connections FROM memories WHERE id = ?", (mid,)).fetchone()
                        if row:
                            existing = json.loads(row["connections"])
                            linked_id = to_id if mid == from_id else from_id
                            # Avoid duplicates
                            if not any(c.get("linked_to") == linked_id for c in existing):
                                existing.append({"type": "memory_link", "linked_to": linked_id, "relationship": rel})
                                db.execute("UPDATE memories SET connections = ? WHERE id = ?", (json.dumps(existing), mid))
            
            elif conn_type == "file_link":
                file_path = conn.get("path")
                if file_path:
                    # Propagate the file_link to every source memory so the
                    # structural linkage is preserved on the originals, not just
                    # the consolidated Insight Cube.
                    link_entry = {
                        "type": "file_link",
                        "path": file_path,
                        "relationship": rel,
                        "status": "active",
                    }
                    for mid in source_ids:
                        try:
                            mid_int = int(mid)
                        except (ValueError, TypeError):
                            continue
                        row = db.execute(
                            "SELECT connections FROM memories WHERE id = ?", (mid_int,)
                        ).fetchone()
                        if row:
                            existing = json.loads(row["connections"])
                            if not any(
                                c.get("type") == "file_link" and c.get("path") == file_path
                                for c in existing
                            ):
                                existing.append(link_entry)
                                db.execute(
                                    "UPDATE memories SET connections = ? WHERE id = ?",
                                    (json.dumps(existing), mid_int),
                                )
        
        # Robust flattening and int-casting for source_ids (LLMs keep returning nested lists)
        flattened_ids = []
        if isinstance(source_ids, list):
            for item in source_ids:
                if isinstance(item, list):
                    flattened_ids.extend(item)
                else:
                    flattened_ids.append(item)
        
        # Ensure all IDs are integers
        clean_ids = []
        for mid in flattened_ids:
            try:
                clean_ids.append(int(mid))
            except (ValueError, TypeError):
                log.warning(f"Skipping invalid source ID: {mid}")
        
        if clean_ids:
            placeholders = ",".join("?" * len(clean_ids))
            db.execute(f"UPDATE memories SET consolidated = 1 WHERE id IN ({placeholders})", clean_ids)
        db.commit()
    
    log.info(f"🔄 Consolidated {len(source_ids)} memories. Insight: {insight[:80]}...")
    return {"status": "consolidated", "memories_processed": len(source_ids), "insight": insight}

def increment_access(memory_id: int):
    """Increment access count and update last_accessed timestamp."""
    with db_session() as db:
        now = datetime.now(timezone.utc).isoformat()
        db.execute(
            "UPDATE memories SET access_count = access_count + 1, last_accessed = ? WHERE id = ?",
            (now, memory_id)
        )
        db.commit()

def export_cubes(memory_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    """Export memories as portable MemCube JSON."""
    with db_session() as db:
        if memory_ids:
            placeholders = ",".join("?" * len(memory_ids))
            rows = db.execute(f"SELECT * FROM memories WHERE id IN ({placeholders})", memory_ids).fetchall()
        else:
            rows = db.execute("SELECT * FROM memories").fetchall()
    
    cubes = []
    for r in rows:
        cube = MemCube(
            cube_id=r["cube_id"],
            sector=r["sector"],
            source=r["source"],
            origin_platform=r["origin_platform"],
            raw_text=r["raw_text"],
            summary=r["summary"],
            entities=json.loads(r["entities"]),
            topics=json.loads(r["topics"]),
            connections=json.loads(r["connections"]),
            metadata=json.loads(r["metadata"]),
            importance_score=r["importance_score"],
            access_count=r["access_count"],
            last_accessed=r["last_accessed"],
            created_at=r["created_at"],
            consolidated=bool(r["consolidated"]),
            valid_to=r["valid_to"]
        )
        cubes.append(cube.export_portable())
    
    return {"cubes": cubes, "count": len(cubes)}

async def import_cubes(cubes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Import MemCube JSON into the database."""
    imported_count = 0
    for cube_data in cubes:
        # Avoid duplicates by cube_id
        with db_session() as db:
            existing = db.execute("SELECT 1 FROM memories WHERE cube_id = ?", (cube_data.get("cube_id"),)).fetchone()
            if existing:
                continue

        # In a real sync we might want to also import embeddings, 
        # but for now we'll re-embed on import if missing
        raw_text = cube_data.get("raw_text", "")
        summary = cube_data.get("summary", "")
        entities = cube_data.get("entities", [])
        topics = cube_data.get("topics", [])
        importance_score = cube_data.get("importance_score", 0.5)
        sector = cube_data.get("sector", "semantic")
        source = cube_data.get("source", "imported")
        origin_platform = cube_data.get("origin_platform", "unknown")
        metadata = cube_data.get("metadata", {})
        
        await store_memory(
            raw_text=raw_text,
            summary=summary,
            entities=entities,
            topics=topics,
            importance_score=importance_score,
            sector=sector,
            source=source,
            origin_platform=origin_platform,
            metadata=metadata,
            cube_id=cube_data.get("cube_id")
        )
        imported_count += 1
        
    return {"status": "success", "imported": imported_count}

def read_consolidation_history(limit: int = 10) -> Dict[str, Any]:
    """Read past consolidation insights."""
    with db_session() as db:
        rows = db.execute("SELECT * FROM consolidations ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        result = [{"summary": r["summary"], "insight": r["insight"], "source_ids": r["source_ids"]} for r in rows]
    return {"consolidations": result, "count": len(result)}

def get_memory_stats() -> Dict[str, Any]:
    """Get current memory statistics."""
    with db_session() as db:
        total = db.execute("SELECT COUNT(*) as c FROM memories").fetchone()["c"]
        unconsolidated = db.execute("SELECT COUNT(*) as c FROM memories WHERE consolidated = 0").fetchone()["c"]
        consolidations = db.execute("SELECT COUNT(*) as c FROM consolidations").fetchone()["c"]
        indexed_docs = db.execute("SELECT COUNT(DISTINCT path) as c FROM documents").fetchone()["c"]
        
        # Count structural links (active vs historical)
        active_links = 0
        historical_links = 0
        rows = db.execute("SELECT connections FROM memories WHERE connections LIKE '%file_link%'").fetchall()
        for r in rows:
            conns = json.loads(r["connections"])
            for c in conns:
                if c.get("type") == "file_link":
                    if c.get("status", "active") == "active":
                        active_links += 1
                    elif c.get("status") == "historical_trace":
                        historical_links += 1
    
    return {
        "total_memories": total,
        "unconsolidated": unconsolidated,
        "consolidations": consolidations,
        "indexed_documents": indexed_docs,
        "structural_links": {
            "active": active_links,
            "historical_trace": historical_links,
            "total": active_links + historical_links
        }
    }

def delete_memory(memory_id: int) -> Dict[str, Any]:
    """Delete a memory by ID."""
    with db_session() as db:
        row = db.execute("SELECT 1 FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            return {"status": "not_found", "memory_id": memory_id}
        db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        if HAS_SQLITE_VEC:
            db.execute("DELETE FROM vec_memories WHERE memory_id = ?", (memory_id,))
        db.commit()
    
    log.info(f"🗑️  Deleted memory #{memory_id}")
    return {"status": "deleted", "memory_id": memory_id}

def update_memory_validity(memory_id: int, valid_to: Optional[str]) -> Dict[str, Any]:
    """Set the valid_to timestamp for an existing memory."""
    with db_session() as db:
        db.execute("UPDATE memories SET valid_to = ? WHERE id = ?", (valid_to, memory_id))
        db.commit()
    log.info(f"⏱️ Updated validity of memory #{memory_id} to {valid_to}")
    return {"status": "updated", "memory_id": memory_id, "valid_to": valid_to}

def reinforce_memory(memory_id: int) -> Dict[str, Any]:
    """Increase the importance of a memory and reset its decay clock."""
    with db_session() as db:
        row = db.execute("SELECT importance_score FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            return {"status": "not_found"}

        new_importance = min(1.0, row["importance_score"] + 0.1)
        now = datetime.now(timezone.utc).isoformat()

        db.execute("UPDATE memories SET importance_score = ?, created_at = ? WHERE id = ?", (new_importance, now, memory_id))
        db.commit()
    log.info(f"💪 Reinforced memory #{memory_id} (Importance: {row['importance_score']:.2f} -> {new_importance:.2f})")
    return {"status": "reinforced", "memory_id": memory_id, "new_importance": new_importance}

def update_link_status(memory_id: int, path: str, new_status: str) -> Dict[str, Any]:
    """
    Update the status of a file_link in a memory's connections.
    Transitions links from 'active' to 'historical_trace' (or other states).
    """
    with db_session() as db:
        row = db.execute("SELECT connections FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            return {"status": "not_found", "memory_id": memory_id}

        connections = json.loads(row["connections"])
        updated = False
        norm_path = os.path.normcase(os.path.abspath(path))
        for conn in connections:
            if conn.get("type") == "file_link" and os.path.normcase(os.path.abspath(conn.get("path", ""))) == norm_path:
                old_status = conn.get("status", "active")
                if old_status != new_status:
                    conn["status"] = new_status
                    conn["updated_at"] = datetime.now(timezone.utc).isoformat()
                    updated = True
        
        if updated:
            db.execute(
                "UPDATE memories SET connections = ? WHERE id = ?", 
                (json.dumps(connections), memory_id)
            )
            db.commit()
            log.info(f"🔗 Updated link status for memory #{memory_id} path '{path}' to '{new_status}'")
            return {"status": "updated", "memory_id": memory_id, "path": path, "new_status": new_status}
        
        return {"status": "no_change", "memory_id": memory_id, "path": path}

def repair_memory(memory_id: int, new_raw_text: str) -> Dict[str, Any]:
    """
    Update a memory's raw_text after drift is detected and a repair is suggested.
    Also resets the created_at timestamp to signify a refresh.
    """
    with db_session() as db:
        row = db.execute("SELECT id FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            return {"status": "not_found", "memory_id": memory_id}

        now = datetime.now(timezone.utc).isoformat()
        db.execute(
            "UPDATE memories SET raw_text = ?, created_at = ? WHERE id = ?", 
            (new_raw_text, now, memory_id)
        )
        db.commit()
        log.info(f"🛠️  Repaired memory #{memory_id} with updated insight.")
        return {"status": "repaired", "memory_id": memory_id}

def get_all_links() -> Dict[str, Any]:
    """Retrieve all file_links from memory connections."""
    links = []
    with db_session() as db:
        rows = db.execute("SELECT id, summary, connections FROM memories WHERE connections LIKE '%file_link%'").fetchall()
        for r in rows:
            conns = json.loads(r["connections"])
            for c in conns:
                if c.get("type") == "file_link":
                    links.append({
                        "memory_id": r["id"],
                        "memory_summary": r["summary"][:60],
                        "path": c.get("path"),
                        "status": c.get("status", "active"),
                        "relationship": c.get("relationship", ""),
                        "updated_at": c.get("updated_at")
                    })
    return {"links": links, "count": len(links)}

def mark_memories_consolidated(memory_ids: List[int]) -> Dict[str, Any]:
    """
    Force-mark a list of memory IDs as consolidated=1.
    Used by the Orphan Sweeper to prevent dropped memories from causing
    infinite re-ingestion loops in subsequent consolidation cycles.

    Args:
        memory_ids: List of integer memory IDs to mark as consolidated.

    Returns:
        Dict with status and count of affected rows.
    """
    if not memory_ids:
        return {"status": "no_op", "marked": 0}

    with db_session() as db:
        placeholders = ",".join("?" * len(memory_ids))
        cursor = db.execute(
            f"UPDATE memories SET consolidated = 1 WHERE id IN ({placeholders})",
            memory_ids
        )
        db.commit()

    count = cursor.rowcount
    log.info(f"🔒 Orphan sweeper: marked {count} memories as consolidated (IDs: {memory_ids}).")
    return {"status": "marked", "marked": count}


def get_memories_by_source(rel_path: str) -> List[int]:
    """Find all memories associated with a specific file path."""
    with db_session() as db:
        # Check direct source field first
        rows = db.execute("SELECT id FROM memories WHERE source = ?", (rel_path,)).fetchall()
        matching_ids = {r["id"] for r in rows}

        # Also find any structural file_links
        rows2 = db.execute("SELECT id, connections FROM memories WHERE connections LIKE ?", (f'%{os.path.basename(rel_path)}%',)).fetchall()
        norm_rel = os.path.normcase(os.path.abspath(rel_path))
        for r in rows2:
            try:
                conns = json.loads(r["connections"])
                for c in conns:
                    if c.get("type") == "file_link" and os.path.normcase(os.path.abspath(c.get("path", ""))) == norm_rel:
                        matching_ids.add(r["id"])
                        break
            except json.JSONDecodeError:
                pass

    return list(matching_ids)



