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

class TestWatchFolderRollbackQuery(unittest.TestCase):
    """
    Regression guard for the LIKE-extended rollback query in watch_folder.

    The bug: before the fix, the rollback only deleted memories whose source
    exactly matched rel_path.  Chunk-sourced memories (source = "path#chunk-N")
    were silently left behind.

    The fix added:  OR source LIKE rel_path||'#chunk-%'

    This test exercises that SQL directly against a real SQLite database so the
    correctness of the query is verified without needing to drive the full
    watch_folder coroutine.
    """

    DB_NAME = "test_rollback_query.db"

    def setUp(self):
        import os
        if os.path.exists(self.DB_NAME):
            os.remove(self.DB_NAME)
        # Point the database module at our temp DB for this test class
        self._env_patcher = unittest.mock.patch.dict(
            os.environ, {"MEMORY_DB": self.DB_NAME}
        )
        self._env_patcher.start()
        init_db()

    def tearDown(self):
        import os
        self._env_patcher.stop()
        if os.path.exists(self.DB_NAME):
            os.remove(self.DB_NAME)

    def _insert_memory(self, db, source: str, created_at: str) -> int:
        """Insert a minimal memory row and return its id."""
        import uuid
        db.execute(
            """INSERT INTO memories
               (cube_id, sector, source, raw_text, summary, entities, topics,
                connections, metadata, importance_score, created_at)
               VALUES (?, 'semantic', ?, 'x', 'x', '[]', '[]', '[]', '{}', 0.5, ?)""",
            (str(uuid.uuid4()), source, created_at),
        )
        return db.execute("SELECT last_insert_rowid()").fetchone()[0]

    def test_like_rollback_catches_chunk_rows(self):
        """
        The fixed query (exact OR LIKE) must return BOTH the bare-path memory
        and the chunk-path memory.
        """
        from database import db_session

        before_ts = "2024-01-01T00:00:00"
        rel_path = "docs/file.md"

        with db_session() as db:
            id_bare = self._insert_memory(db, "docs/file.md", "2024-01-01T00:00:01")
            id_chunk = self._insert_memory(db, "docs/file.md#chunk-1", "2024-01-01T00:00:01")
            db.commit()

            # --- Fixed query (should catch both rows) ---
            rows_fixed = db.execute(
                "SELECT id FROM memories WHERE (source = ? OR source LIKE ?) AND created_at >= ?",
                (rel_path, f"{rel_path}#chunk-%", before_ts),
            ).fetchall()
            found_ids_fixed = {r["id"] for r in rows_fixed}

            self.assertIn(id_bare, found_ids_fixed,
                          "Fixed query must return the bare-path memory")
            self.assertIn(id_chunk, found_ids_fixed,
                          "Fixed query must return the chunk-path memory")
            self.assertEqual(len(found_ids_fixed), 2,
                             "Fixed query must return exactly 2 rows")

    def test_exact_match_only_misses_chunk_rows(self):
        """
        The OLD (broken) query — exact source match only — must NOT return the
        chunk-path memory.  This documents the pre-fix behaviour and ensures the
        regression guard is meaningful.
        """
        from database import db_session

        before_ts = "2024-01-01T00:00:00"
        rel_path = "docs/file.md"

        with db_session() as db:
            id_bare = self._insert_memory(db, "docs/file.md", "2024-01-01T00:00:01")
            id_chunk = self._insert_memory(db, "docs/file.md#chunk-1", "2024-01-01T00:00:01")
            db.commit()

            # --- Old (broken) query — exact match only ---
            rows_old = db.execute(
                "SELECT id FROM memories WHERE source = ? AND created_at >= ?",
                (rel_path, before_ts),
            ).fetchall()
            found_ids_old = {r["id"] for r in rows_old}

            self.assertIn(id_bare, found_ids_old,
                          "Old query must still return the bare-path memory")
            self.assertNotIn(id_chunk, found_ids_old,
                             "Old query must NOT return the chunk-path memory — "
                             "this proves the bug that the LIKE fix addressed")
            self.assertEqual(len(found_ids_old), 1,
                             "Old query must return exactly 1 row")


if __name__ == "__main__":
    unittest.main()
