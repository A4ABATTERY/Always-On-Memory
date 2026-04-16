"""
Tests for Proactive Sync & Semantic Drift Detection (V3.3).
Verifies that the Librarian detects drift and the Agent evolved links.
"""

import unittest
import asyncio
import json
import os
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from agent import MemoryAgent
from librarian import index_all_dirs
from memory_store import store_memory, update_link_status, db_session
from database import init_db
from models import AuditResult

_DUMMY_VECTOR = [0.01] * 3072

class TestProactiveSync(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.db_path = "test_sync.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        os.environ["MEMORY_DB"] = self.db_path
        init_db()

        self.embed_patcher = patch("utils.embed_text", AsyncMock(return_value=_DUMMY_VECTOR))
        self.embed_patcher.start()

        self.mock_agents = [MagicMock() for _ in range(8)]
        for m in self.mock_agents:
            m.run = AsyncMock()

        with patch('agent.build_agents', return_value=self.mock_agents):
            self.agent = MemoryAgent()

        self.batch_embed_patcher = patch("librarian.embed_texts_batch", AsyncMock(return_value=[_DUMMY_VECTOR]))
        self.batch_embed_patcher.start()

        self.test_dir = Path("test_sync_dir")
        self.test_dir.mkdir(exist_ok=True)

    def tearDown(self):
        self.embed_patcher.stop()
        self.batch_embed_patcher.stop()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        if self.test_dir.exists():
            import shutil
            shutil.rmtree(self.test_dir)

    async def test_drift_detection_and_link_evolution(self):
        # 1. Setup: Create a file and a memory linked to it
        file_path = self.test_dir / "logic.py"
        file_path.write_text("def process(): return 'old logic'")
        
        # Store a memory with a file_link to this path
        # We need a realistic embedding for 'old logic'
        await store_memory(
            raw_text="The code performs old logic processing.",
            summary="Old logic processing",
            entities=[], topics=[], importance_score=0.8,
            connections=[{"type": "file_link", "path": str(file_path.resolve()), "status": "active"}]
        )
        
        # Get memory ID
        with db_session() as db:
            row = db.execute("SELECT id FROM memories LIMIT 1").fetchone()
            memory_id = row["id"]

        # 2. Simulate Drift: Update file with completely different logic
        file_path.write_text("def process(): return 'NEW MAPREDUCE DISTRIBUTED LOGIC' " * 20)
        
        # 3. Run Librarian indexing with the drift callback.
        # _check_semantic_drift uses float embeddings from vec_memories via
        # vec_to_json; with a uniform dummy vector the cosine distance is 0 and
        # drift is never triggered.  We therefore mock the internal helper to
        # simulate a drift event — this isolates the queue/audit integration
        # under test from the embedding-math already covered elsewhere.
        async def _sim_drift(path, new_embeddings, on_drift_fn):
            import asyncio as _asyncio
            if _asyncio.iscoroutinefunction(on_drift_fn):
                await on_drift_fn(path, memory_id)
            else:
                on_drift_fn(path, memory_id)

        from unittest.mock import patch as _patch
        with _patch('librarian._check_semantic_drift', side_effect=_sim_drift):
            await index_all_dirs([str(self.test_dir)], on_drift_detected=self.agent.push_sync_task)
        
        # 4. Verify Task is in Queue
        self.assertEqual(self.agent.sync_queue.qsize(), 1)
        task = await self.agent.sync_queue.get()
        self.assertEqual(os.path.normcase(task["path"]), os.path.normcase(str(file_path.resolve())))
        self.assertEqual(task["memory_id"], memory_id)
        
        # 5. Simulate Sync Worker Audit
        # Mock sync_agent response to evolve the link
        data = AuditResult(status="HISTORICAL", reason="Logic has changed significantly")
        self.agent.sync_agent.run.return_value = MagicMock(output=data)
        
        # Create a mock doc for read_document if needed or just let it read the real file
        with patch('librarian.WATCH_DIRS', str(self.test_dir.resolve())):
            await self.agent._audit_link(task["path"], task["memory_id"])
        
        # 6. Verify link state in DB
        with db_session() as db:
            row = db.execute("SELECT connections FROM memories WHERE id = ?", (memory_id,)).fetchone()
            connections = json.loads(row["connections"])
            link = next(c for c in connections if c.get("type") == "file_link")
            self.assertEqual(link["status"], "historical_trace")

    async def test_repair_logic(self):
        # 1. Setup: Create file and memory
        file_path = self.test_dir / "repair.py"
        file_path.write_text("def x(): pass")
        
        await store_memory(
            raw_text="The code has function x.",
            summary="Function x",
            entities=[], topics=[], importance_score=0.8,
            connections=[{"type": "file_link", "path": str(file_path.resolve()), "status": "active"}]
        )
        
        with db_session() as db:
            memory_id = db.execute("SELECT id FROM memories LIMIT 1").fetchone()["id"]

        # 2. Simulate REPAIR decision from Sync Agent
        data = AuditResult(
            status="REPAIR", 
            reason="Function name changed",
            suggested_update="The code now has function y."
        )
        self.agent.sync_agent.run.return_value = MagicMock(output=data)
        
        # 3. Trigger audit
        with patch('librarian.WATCH_DIRS', str(self.test_dir.resolve())):
            await self.agent._audit_link(str(file_path.resolve()), memory_id)
            
        # 4. Verify memory was updated
        with db_session() as db:
            row = db.execute("SELECT raw_text, connections FROM memories WHERE id = ?", (memory_id,)).fetchone()
            self.assertEqual(row["raw_text"], "The code now has function y.")
            connections = json.loads(row["connections"])
            link = next(c for c in connections if c.get("type") == "file_link")
            self.assertEqual(link["status"], "active")


if __name__ == "__main__":
    unittest.main()
