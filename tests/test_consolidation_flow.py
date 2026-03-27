"""
Integration tests for the full memory consolidation flow in Always-On-Memory.
Verifies that agents_factory, agent, and memory_store work together to persist consolidations.
"""

import unittest
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
from datetime import datetime, timezone

from agent import MemoryAgent
from database import init_db, db_session
from memory_store import store_memory
from models import SynthesisResult, EvalResult

class TestConsolidationFlow(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        # Use a temporary test database
        self.db_path = "test_flow_v3.db"
        self.env_patcher = unittest.mock.patch.dict(os.environ, {"MEMORY_DB": self.db_path})
        self.env_patcher.start()
        
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
            
        init_db()
        
        # Mock the agents to avoid real LLM calls
        self.mock_agents = [MagicMock() for _ in range(8)]
        for m in self.mock_agents:
            m.run = AsyncMock()
            
        with patch('agent.build_agents', return_value=self.mock_agents):
            self.agent = MemoryAgent()

    def tearDown(self):
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        self.env_patcher.stop()

    async def test_full_consolidation_and_persistence(self):
        """Test that consolidation correctly saves records and marks memories as consolidated."""
        # 1. Seed some unconsolidated memories
        m1 = await store_memory("Fact 1: London is cold", "London cold", [], ["weather"], 0.5)
        m2 = await store_memory("Fact 2: London has a bridge", "London bridge", [], ["infrastructure"], 0.5)
        
        # Verify they are unconsolidated
        with db_session() as db:
            count = db.execute("SELECT COUNT(*) as c FROM memories WHERE consolidated = 0").fetchone()["c"]
            self.assertEqual(count, 2)

        # 2. Mock Agent Responses
        # Generator synthesis
        gen_data = SynthesisResult(
            summary="London summary",
            insight="London is cold and has a bridge.",
            source_ids=[m1["memory_id"], m2["memory_id"]],
            connections=[{"from_id": m1["memory_id"], "to_id": m2["memory_id"], "relationship": "same_city"}]
        )
        self.agent.generator_lite.run.return_value = MagicMock(data=gen_data)
        
        # Evaluator approval
        eval_data = EvalResult(
            score=0.9,
            feedback="Good",
            fidelity=1.0,
            completeness=1.0,
            redundancy_removed=1.0
        )
        self.agent.evaluator_lite.run.return_value = MagicMock(data=eval_data)

        # 3. Trigger Consolidation
        msg = await self.agent.consolidate()
        self.assertIn("Consolidated 2 memories into Insight Cube", msg)

        # 4. Verify Database State
        with db_session() as db:
            # Check that old memories are marked consolidated
            unconsolidated_count = db.execute("SELECT COUNT(*) as c FROM memories WHERE consolidated = 0").fetchone()["c"]
            # We expect 1 (the NEW Insight cube)
            self.assertEqual(unconsolidated_count, 1)
            
            # Check that 2 are marked as consolidated
            consolidated_count = db.execute("SELECT COUNT(*) as c FROM memories WHERE consolidated = 1").fetchone()["c"]
            self.assertEqual(consolidated_count, 2)
            
            # Check Consolidation record
            cons = db.execute("SELECT * FROM consolidations").fetchone()
            self.assertIsNotNone(cons)
            self.assertEqual(cons["summary"], "London summary")
            
            # Check the new Insight Cube
            insight_cube = db.execute("SELECT * FROM memories WHERE topics LIKE '%consolidated-insight%'").fetchone()
            self.assertIsNotNone(insight_cube)
            self.assertEqual(insight_cube["raw_text"], "London is cold and has a bridge.")
            self.assertEqual(insight_cube["source"], "adversarial-consolidation")

if __name__ == "__main__":
    unittest.main()
