"""
Edge case tests for Always-On-Memory Agent.
"""

import os
import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch, MagicMock

# Global test DB name
TEST_DB = "test_edges_v3.db"
_DUMMY_VECTOR = [0.01] * 3072

from database import init_db
from memory_store import store_memory, get_memory_stats, delete_memory

class TestMemoryEdges(unittest.IsolatedAsyncioTestCase):

    @classmethod
    def setUpClass(cls):
        cls.env_patcher = unittest.mock.patch.dict(os.environ, {"MEMORY_DB": TEST_DB})
        cls.env_patcher.start()
        cls.embed_patcher = patch("utils.embed_text", AsyncMock(return_value=_DUMMY_VECTOR))
        cls.embed_patcher.start()

        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)
        init_db()

    @classmethod
    def tearDownClass(cls):
        cls.embed_patcher.stop()
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

    async def test_nested_folder_watcher(self):
        """Verify that files within nested folders are successfully processed."""
        import tempfile
        import asyncio
        from pathlib import Path
        from agent import watch_folder
        import agent
        
        agent._shutdown_event = asyncio.Event()
        mock_agent = MagicMock()
        mock_agent.ingest = AsyncMock(return_value="Ingested")

        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            nested = folder / "sub" / "folder"
            nested.mkdir(parents=True)
            test_file = nested / "nested_test.txt"
            test_file.write_bytes(b"content")

            watch_task = asyncio.create_task(watch_folder(mock_agent, folder, poll_interval=1))
            await asyncio.sleep(0.5)
            agent._shutdown_event.set()
            try:
                await asyncio.wait_for(watch_task, timeout=2.0)
            except asyncio.CancelledError:
                pass
            
            self.assertEqual(mock_agent.ingest.call_count, 1)
            
            from database import db_session
            with db_session() as db:
                row = db.execute("SELECT 1 FROM processed_files WHERE path = ?", ("sub/folder/nested_test.txt",)).fetchone()
                self.assertIsNotNone(row)

if __name__ == "__main__":
    unittest.main()
