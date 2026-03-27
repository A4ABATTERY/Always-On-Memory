"""
Edge case tests for Always-On-Memory Agent.
"""

import os
import unittest
from datetime import datetime, timezone

# Global test DB name
TEST_DB = "test_edges_v3.db"

from database import init_db
from memory_store import store_memory, get_memory_stats, delete_memory

class TestMemoryEdges(unittest.IsolatedAsyncioTestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.env_patcher = unittest.mock.patch.dict(os.environ, {"MEMORY_DB": TEST_DB})
        cls.env_patcher.start()
        
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)
        init_db()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)
        cls.env_patcher.stop()

    async def test_empty_raw_text(self):
        """Test behavior when raw_text is empty."""
        res = await store_memory(
            raw_text="", summary="Empty", entities=[], topics=[], importance_score=0.1
        )
        self.assertIn("memory_id", res)
        stats = get_memory_stats()
        self.assertGreaterEqual(stats["total_memories"], 1)

    async def test_extremely_large_text(self):
        """Test behavior with large payload (1MB)."""
        large_text = "A" * (1024 * 1024)
        res = await store_memory(
            raw_text=large_text, summary="Large", entities=[], topics=[], importance_score=0.5
        )
        self.assertIn("memory_id", res)
        # Cleanup
        delete_memory(res["memory_id"])

    async def test_special_characters(self):
        """Test emojis and SQL-like strings."""
        special = "🤖 🚀 ' OR 1=1; -- DROP TABLE memories;"
        res = await store_memory(
            raw_text=special, summary="Special", entities=["🤖"], topics=["sql"], importance_score=0.9
        )
        self.assertIn("memory_id", res)
        
        from database import db_session
        with db_session() as db:
            row = db.execute("SELECT raw_text FROM memories WHERE id = ?", (res["memory_id"],)).fetchone()
            self.assertEqual(row["raw_text"], special)

if __name__ == "__main__":
    unittest.main()
