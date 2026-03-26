"""
Memory Store Module — Encapsulates database operations for memories and consolidations.
"""

import json
import logging
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

from database import db_session
from models import Memory, Consolidation

log = logging.getLogger("memory-agent.store")

def store_memory(
    raw_text: str,
    summary: str,
    entities: List[str],
    topics: List[str],
    importance: float,
    sector: str = "semantic",
    valid_to: Optional[str] = None,
    source: str = "",
) -> Dict[str, Any]:
    """Store a processed memory in the database."""
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
            "id": r["id"], "summary": r["summary"],
            "entities": json.loads(r["entities"]), "topics": json.loads(r["topics"]),
            "importance": r["importance"], "created_at": r["created_at"],
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
            "id": r["id"], "summary": r["summary"], "raw_text": r["raw_text"],
            "importance": r["importance"], "created_at": r["created_at"],
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
                        existing.append({"linked_to": to_id if mid == from_id else from_id, "relationship": rel})
                        db.execute("UPDATE memories SET connections = ? WHERE id = ?", (json.dumps(existing), mid))
        
        placeholders = ",".join("?" * len(source_ids))
        db.execute(f"UPDATE memories SET consolidated = 1 WHERE id IN ({placeholders})", source_ids)
        db.commit()
    
    log.info(f"🔄 Consolidated {len(source_ids)} memories. Insight: {insight[:80]}...")
    return {"status": "consolidated", "memories_processed": len(source_ids), "insight": insight}

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
        row = db.execute("SELECT importance FROM memories WHERE id = ?", (memory_id,)).fetchone()
        if not row:
            return {"status": "not_found"}

        new_importance = min(1.0, row["importance"] + 0.1)
        now = datetime.now(timezone.utc).isoformat()

        db.execute("UPDATE memories SET importance = ?, created_at = ? WHERE id = ?", (new_importance, now, memory_id))
        db.commit()
    log.info(f"💪 Reinforced memory #{memory_id} (Importance: {row['importance']:.2f} -> {new_importance:.2f})")
    return {"status": "reinforced", "memory_id": memory_id, "new_importance": new_importance}
