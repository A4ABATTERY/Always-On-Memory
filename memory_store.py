"""
Memory Store Module — Encapsulates database operations for memories and consolidations.
"""

import json
import logging
from uuid import uuid4
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from database import db_session
from models import MemCube, Consolidation
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
    embedding: Optional[List[float]] = None,
    cube_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Store a MemCube in the database."""
    if embedding is None:
        embedding = await embed_text(raw_text) or []

    with db_session() as db:
        now = datetime.now(timezone.utc).isoformat()
        cube = MemCube(
            cube_id=cube_id or str(uuid4()) if 'uuid4' in globals() else None, # Handled by Field factory
            raw_text=raw_text,
            summary=summary,
            entities=entities,
            topics=topics,
            importance_score=importance_score,
            sector=sector,
            valid_to=valid_to,
            source=source,
            origin_platform=origin_platform,
            metadata=metadata or {},
            created_at=now,
        )
        
        # Override cube_id if provided explicitly (MemCube factory is just for defaults)
        if cube_id:
            # Re-create to ensure immutability is respected if frozen, 
            # but actually MemCube uses Field(default_factory=...)
            # The cleanest way is to pass it to the constructor.
            cube = MemCube(
                cube_id=cube_id,
                raw_text=raw_text,
                summary=summary,
                entities=entities,
                topics=topics,
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
                    "INSERT INTO vec_memories (memory_id, embedding) VALUES (?, vec_quantize_int8(vec_f32(?), 'unit'))",
                    (mid, json.dumps(embedding)),
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
    """Read memories that haven't been consolidated yet."""
    with db_session() as db:
        rows = db.execute(
            "SELECT * FROM memories WHERE consolidated = 0 ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    memories = []
    for r in rows:
        memories.append({
            "id": r["id"], "cube_id": r["cube_id"], "summary": r["summary"],
            "entities": json.loads(r["entities"]), "topics": json.loads(r["topics"]),
            "importance_score": r["importance_score"], "created_at": r["created_at"],
        })
    return {"memories": memories, "count": len(memories)}

def read_memory_partition(sector: str, limit: int = 100) -> Dict[str, Any]:
    """Fetches memories from a specific sector."""
    with db_session() as db:
        rows = db.execute(
            "SELECT * FROM memories WHERE sector = ? ORDER BY created_at DESC LIMIT ?",
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
            from_id, to_id = conn.get("from_id"), conn.get("to_id")
            rel = conn.get("relationship", "")
            if from_id and to_id:
                for mid in [from_id, to_id]:
                    row = db.execute("SELECT connections FROM memories WHERE id = ?", (mid,)).fetchone()
                    if row:
                        existing = json.loads(row["connections"])
                        linked_id = to_id if mid == from_id else from_id
                        # Avoid duplicates
                        if not any(c.get("linked_to") == linked_id for c in existing):
                            existing.append({"linked_to": linked_id, "relationship": rel})
                            db.execute("UPDATE memories SET connections = ? WHERE id = ?", (json.dumps(existing), mid))
        
        if source_ids:
            placeholders = ",".join("?" * len(source_ids))
            db.execute(f"UPDATE memories SET consolidated = 1 WHERE id IN ({placeholders})", source_ids)
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
    
    return {
        "total_memories": total,
        "unconsolidated": unconsolidated,
        "consolidations": consolidations,
        "indexed_documents": indexed_docs,
    }

def delete_memory(memory_id: int) -> Dict[str, Any]:
    """Delete a memory by ID."""
    with db_session() as db:
        row = db.execute("SELECT 1 FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            return {"status": "not_found", "memory_id": memory_id}
        db.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        db.execute("DELETE FROM vec_memories WHERE memory_id = ?", (memory_id,))
        db.commit()
    
    log.info(f"🗑️  Deleted memory #{memory_id}")
    return {"status": "deleted", "memory_id": memory_id}

def update_memory_validity(memory_id: int, valid_to: str) -> Dict[str, Any]:
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
