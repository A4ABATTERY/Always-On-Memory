"""
Unittest suite for Always-On-Memory Agent.
"""

import os
import unittest
import json
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

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

    async def test_04_watch_folder_hash_update(self):
        """Verify that updating a file triggers hash change logic."""
        import tempfile
        import asyncio
        import hashlib
        from agent import watch_folder
        import agent
        
        agent._shutdown_event = asyncio.Event()
        
        mock_agent = MagicMock()
        mock_agent.ingest = AsyncMock(return_value="Ingested")

        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            test_file = folder / "update_test.txt"
            
            test_file.write_text("v1 content")
            
            watch_task = asyncio.create_task(watch_folder(mock_agent, folder, poll_interval=1))
            await asyncio.sleep(0.5)
            agent._shutdown_event.set()
            # Catch cancellation
            try:
                await asyncio.wait_for(watch_task, timeout=2.0)
            except asyncio.CancelledError:
                pass
            
            self.assertEqual(mock_agent.ingest.call_count, 1)
            
            with db_session() as db:
                row = db.execute("SELECT content_hash FROM processed_files WHERE path = ?", ("update_test.txt",)).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row["content_hash"], hashlib.md5(b"v1 content").hexdigest())
                
            # Second run with updated file
            agent._shutdown_event.clear()
            test_file.write_text("v2 content")
            watch_task = asyncio.create_task(watch_folder(mock_agent, folder, poll_interval=1))
            await asyncio.sleep(0.5)
            agent._shutdown_event.set()
            try:
                await asyncio.wait_for(watch_task, timeout=2.0)
            except asyncio.CancelledError:
                pass
            
            self.assertEqual(mock_agent.ingest.call_count, 2)
            
            with db_session() as db:
                row = db.execute("SELECT content_hash, prev_hash FROM processed_files WHERE path = ?", ("update_test.txt",)).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row["content_hash"], hashlib.md5(b"v2 content").hexdigest())
                self.assertEqual(row["prev_hash"], hashlib.md5(b"v1 content").hexdigest())

if __name__ == "__main__":
    # Note: test_01 is async, so we'd need to run it with an event loop if using standard unittest
    # But for simplicity, let's just use pytest which handles async tests better.
    unittest.main()
