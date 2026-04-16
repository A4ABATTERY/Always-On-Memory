"""
Unit tests for the Generator-Evaluator adversarial loop.
"""

import unittest
import json
from unittest.mock import AsyncMock, MagicMock, patch
from agent import MemoryAgent
from models import MultiSynthesisResult, TopicSynthesis, EvalResult

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
        gen_data = MultiSynthesisResult(
            insights=[
                TopicSynthesis(
                    topic_name="Test Topic",
                    summary="Good summary",
                    insight="Deep insight",
                    source_ids=[1, 2],
                    connections=[]
                )
            ]
        )
        self.agent.generator_lite.run.return_value = MagicMock(output=gen_data)
        eval_data = EvalResult(
            score=0.95,
            feedback="Perfect",
            fidelity=1.0,
            source_coverage=1.0,
            topic_cohesion=1.0,
            redundancy_removed=1.0
        )
        self.agent.evaluator_lite.run.return_value = MagicMock(output=eval_data)

        result = await self.agent.adversarial_consolidation(
            self.agent.generator_lite,
            self.agent.evaluator_lite,
            "Raw memories"
        )

        self.assertIsInstance(result, dict)
        self.assertIn("insights", result)
        self.assertEqual(result["insights"][0]["summary"], "Good summary")
        self.assertEqual(self.agent.generator_lite.run.call_count, 1)
        self.assertEqual(self.agent.evaluator_lite.run.call_count, 1)

    async def test_adversarial_retry_logic(self):
        """Test that the loop retries when scores are low."""
        gen_data = MultiSynthesisResult(
            insights=[
                TopicSynthesis(
                    topic_name="Draft Topic",
                    summary="Draft summary",
                    insight="Draft insight",
                    source_ids=[1, 2],
                    connections=[]
                )
            ]
        )
        self.agent.generator_lite.run.return_value = MagicMock(output=gen_data)

        eval_1 = EvalResult(score=0.5, feedback="Too short", fidelity=0.5, source_coverage=0.5, topic_cohesion=0.5, redundancy_removed=0.5)
        eval_2 = EvalResult(score=0.9, feedback="Better", fidelity=0.9, source_coverage=0.9, topic_cohesion=0.9, redundancy_removed=0.9)

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

        self.assertEqual(result["insights"][0]["summary"], "Draft summary")
        self.assertEqual(self.agent.generator_lite.run.call_count, 2)
        self.assertEqual(self.agent.evaluator_lite.run.call_count, 2)

    async def test_adversarial_failure_max_attempts(self):
        """Test that it raises an exception after max_attempts."""
        gen_data = MultiSynthesisResult(
            insights=[
                TopicSynthesis(
                    topic_name="Bad Topic",
                    summary="Bad summary",
                    insight="Bad insight",
                    source_ids=[1, 2],
                    connections=[]
                )
            ]
        )
        self.agent.generator_lite.run.return_value = MagicMock(output=gen_data)
        self.agent.evaluator_lite.run.return_value = MagicMock(output=EvalResult(score=0.4, feedback="Bad", fidelity=0.4, source_coverage=0.4, topic_cohesion=0.4, redundancy_removed=0.4))

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

if __name__ == "__main__":
    unittest.main()
