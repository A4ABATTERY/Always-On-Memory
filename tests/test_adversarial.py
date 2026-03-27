"""
Unit tests for the Generator-Evaluator adversarial loop.
"""

import unittest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from agent import MemoryAgent

class TestAdversarialLoop(unittest.IsolatedAsyncioTestCase):
    
    def setUp(self):
        # Patch build_agents to return mocks
        self.mock_agents = [MagicMock() for _ in range(8)]
        for m in self.mock_agents:
            m.run = AsyncMock()
            
        with patch('agent.build_agents', return_value=self.mock_agents):
            with patch('agent.init_db'):
                self.agent = MemoryAgent()

    async def test_adversarial_success_first_try(self):
        """Test that consolidation succeeds immediately if score is high enough."""
        # Generator result
        gen_output = json.dumps({
            "summary": "Good summary",
            "insight": "Deep insight",
            "source_ids": [1, 2],
            "connections": []
        })
        self.agent.generator_lite.run.return_value = MagicMock(output=gen_output)
        # Evaluator result
        eval_output = json.dumps({
            "score": 0.95,
            "feedback": "Perfect",
            "fidelity": 1.0,
            "completeness": 1.0,
            "redundancy_removed": 1.0
        })
        self.agent.evaluator_lite.run.return_value = MagicMock(output=eval_output)
        
        result = await self.agent.adversarial_consolidation(
            self.agent.generator_lite,
            self.agent.evaluator_lite,
            "Raw memories"
        )
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result["summary"], "Good summary")
        self.assertEqual(self.agent.generator_lite.run.call_count, 1)
        self.assertEqual(self.agent.evaluator_lite.run.call_count, 1)

    async def test_adversarial_retry_logic(self):
        """Test that the loop retries when scores are low."""
        # Generator always returns a summary
        gen_output = json.dumps({
            "summary": "Draft summary",
            "insight": "Draft insight",
            "source_ids": [1, 2],
            "connections": []
        })
        self.agent.generator_lite.run.return_value = MagicMock(output=gen_output)
        
        # Evaluator returns 0.5 then 0.9
        eval_1 = json.dumps({"score": 0.5, "feedback": "Too short"})
        eval_2 = json.dumps({"score": 0.9, "feedback": "Better"})
        
        self.agent.evaluator_lite.run.side_effect = [
            MagicMock(output=eval_1),
            MagicMock(output=eval_2)
        ]
        
        result = await self.agent.adversarial_consolidation(
            self.agent.generator_lite,
            self.agent.evaluator_lite,
            "Raw memories",
            max_attempts=3,
            quality_threshold=0.8
        )
        
        self.assertEqual(result["summary"], "Draft summary")
        self.assertEqual(self.agent.generator_lite.run.call_count, 2)
        self.assertEqual(self.agent.evaluator_lite.run.call_count, 2)

    async def test_adversarial_failure_max_attempts(self):
        """Test that it raises an exception after max_attempts."""
        gen_output = json.dumps({
            "summary": "Bad summary",
            "insight": "Bad insight",
            "source_ids": [1, 2],
            "connections": []
        })
        self.agent.generator_lite.run.return_value = MagicMock(output=gen_output)
        self.agent.evaluator_lite.run.return_value = MagicMock(output=json.dumps({"score": 0.4}))
        
        with self.assertRaises(Exception) as cm:
            await self.agent.adversarial_consolidation(
                self.agent.generator_lite,
                self.agent.evaluator_lite,
                "Raw memories",
                max_attempts=2,
                quality_threshold=0.8
            )
        self.assertIn("Consolidation quality threshold not met", str(cm.exception))
        self.assertEqual(self.agent.generator_lite.run.call_count, 2)

    async def test_json_markdown_stripping(self):
        """Test that markdown-wrapped JSON is correctly parsed."""
        gen_output = "```json\n{\"summary\": \"Summary\", \"insight\": \"I\", \"source_ids\": [], \"connections\": []}\n```"
        self.agent.generator_lite.run.return_value = MagicMock(output=gen_output)
        
        # Wrapped in markdown
        eval_output = "```json\n{\"score\": 0.9, \"feedback\": \"ok\"}\n```"
        self.agent.evaluator_lite.run.return_value = MagicMock(output=eval_output)
        
        result = await self.agent.adversarial_consolidation(
            self.agent.generator_lite,
            self.agent.evaluator_lite,
            "Raw memories"
        )
        self.assertEqual(result["summary"], "Summary")

    async def test_reject_string_json(self):
        """Test that it rejects a JSON string and retries until it gets an object."""
        # First attempt: JSON string
        gen_output_1 = json.dumps("I am just a string")
        # Second attempt: Correct JSON object
        gen_output_2 = json.dumps({"summary": "Actually an object", "insight": "I", "source_ids": [], "connections": []})
        
        self.agent.generator_lite.run.side_effect = [
            MagicMock(output=gen_output_1),
            MagicMock(output=gen_output_2)
        ]
        
        eval_output = json.dumps({"score": 0.9, "feedback": "ok"})
        self.agent.evaluator_lite.run.return_value = MagicMock(output=eval_output)
        
        result = await self.agent.adversarial_consolidation(
            self.agent.generator_lite,
            self.agent.evaluator_lite,
            "Raw memories"
        )
        self.assertEqual(result["summary"], "Actually an object")
        self.assertEqual(self.agent.generator_lite.run.call_count, 2)

if __name__ == "__main__":
    unittest.main()
