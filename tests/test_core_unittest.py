"""
Unittest suite for Always-On-Memory Agent.
"""

import os
import unittest
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

# Global test DB name
TEST_DB = "test_core_v3.db"
_DUMMY_VECTOR = [0.01] * 3072

from database import init_db, db_session
from memory_store import store_memory, read_all_memories, get_memory_stats
from models import MemCube

class TestMemoryAgent(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        """Initialize a fresh test database for each test."""
        self.env_patcher = unittest.mock.patch.dict(os.environ, {"MEMORY_DB": TEST_DB})
        self.env_patcher.start()
        # Prevent real Gemini API calls during store_memory
        self.embed_patcher = patch("utils.embed_text", AsyncMock(return_value=_DUMMY_VECTOR))
        self.embed_patcher.start()

        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)
        init_db()

    def tearDown(self):
        """Clean up the test database."""
        self.embed_patcher.stop()
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)
        self.env_patcher.stop()

    async def test_01_store_and_read(self):
        """Test basic storage and retrieval."""
        await store_memory(
            raw_text="Test memory content",
            summary="Test summary",
            entities=["test"],
            topics=["testing"],
            importance_score=0.8,
            sector="semantic"
        )
        
        stats = get_memory_stats()
        self.assertEqual(stats["total_memories"], 1)
        
        mems = read_all_memories()
        self.assertEqual(mems["count"], 1)
        self.assertEqual(mems["memories"][0]["summary"], "Test summary")
        self.assertEqual(mems["memories"][0]["importance_score"], 0.8)

    def test_02_immutability(self):
        """Verify that MemCube model is immutable."""
        m = MemCube(
            raw_text="abc", summary="def", created_at=datetime.now(timezone.utc).isoformat(),
            importance_score=0.5, entities=[], topics=[]
        )
        # Pydantic v2 models with frozen=True raise ValidationError or AttributeError
        with self.assertRaises((AttributeError, Exception)):
            m.importance_score = 0.9

    def test_03_composite_scoring(self):
        """Test that score calculations rank correctly."""
        now = datetime.now(timezone.utc)
        old_time = (now - timedelta(days=2)).isoformat()
        
        with db_session() as db:
            db.execute(
                """INSERT INTO memories (
                    cube_id, raw_text, summary, entities, topics, connections, 
                    metadata, importance_score, created_at
                ) VALUES (?, ?, ?, '[]', '[]', '[]', '{}', ?, ?)""",
                ("old-cube", "Old", "Old Summary", 0.9, old_time)
            )
            db.execute(
                """INSERT INTO memories (
                    cube_id, raw_text, summary, entities, topics, connections, 
                    metadata, importance_score, created_at
                ) VALUES (?, ?, ?, '[]', '[]', '[]', '{}', ?, ?)""",
                ("new-cube", "New", "New Summary", 0.9, now.isoformat())
            )
            db.commit()
        
        mems = read_all_memories()
        # Find the new and old ones specifically
        new_mem = next(m for m in mems["memories"] if m["summary"] == "New Summary")
        old_mem = next(m for m in mems["memories"] if m["summary"] == "Old Summary")
        
        self.assertGreater(new_mem["composite_score"], old_mem["composite_score"])

if __name__ == "__main__":
    # Note: test_01 is async, so we'd need to run it with an event loop if using standard unittest
    # But for simplicity, let's just use pytest which handles async tests better.
    unittest.main()
