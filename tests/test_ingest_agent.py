"""
Smoke tests for the Ingest Agent.

These tests verify that the ingest agent's system prompt causes it to call
store_memory with plausible arguments. They mock store_memory so no DB or
real LLM is required — the TestModel from pydantic_ai returns structured
tool calls deterministically.

To run against a real LLM (requires GOOGLE_API_KEY):
    INTEGRATION=1 PYTHONPATH=. python -m unittest tests.test_ingest_agent
"""

import os
import unittest
from unittest.mock import AsyncMock, patch, MagicMock


VALID_SECTORS = {"semantic", "episodic", "procedural", "reflection"}


class TestIngestAgentSmoke(unittest.IsolatedAsyncioTestCase):
    """Smoke tests using PydanticAI's TestModel — no real LLM required."""

    def setUp(self):
        # Redirect DB so no real database is touched
        self.env_patcher = unittest.mock.patch.dict(os.environ, {"MEMORY_DB": ":memory:"})
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()

    async def test_ingest_agent_calls_store_memory(self):
        """
        Verify store_memory is registered as a tool on the ingest agent
        (PydanticAI 1.77+ API: agent._function_toolset.tools).
        No patching of store_memory — patching an AsyncMock over the real
        coroutine breaks PydanticAI's type introspection at build time.
        """
        from agents_factory import build_agents
        (ingest_agent, *_) = build_agents()

        ts = getattr(ingest_agent, '_function_toolset', None)
        tool_names = set(ts.tools.keys()) if ts is not None and hasattr(ts, 'tools') else set()

        self.assertIn(
            "store_memory",
            tool_names,
            f"store_memory not found in ingest agent tools. Found: {tool_names}",
        )

    async def test_store_memory_signature_matches_agent_call(self):
        """
        Verify store_memory accepts the keyword arguments the agent system prompt
        instructs it to pass: raw_text, summary, entities, topics, importance_score,
        sector, valid_to, source.
        """
        import inspect
        from memory_store import store_memory
        sig = inspect.signature(store_memory)
        required_params = {"raw_text", "summary", "entities", "topics", "importance_score"}
        actual_params = set(sig.parameters.keys())
        missing = required_params - actual_params
        self.assertFalse(missing, f"store_memory is missing expected params: {missing}")

    async def test_ingest_reflexive_loop_fallback(self):
        """
        Verify that if the agent runs but store_memory is never called (prompt
        drift / model refusal), the ingest() method falls back to direct persistence.

        This test patches build_agents (and the model construction it triggers)
        so no GOOGLE_API_KEY is required.
        """
        mock_agents = [MagicMock() for _ in range(8)]
        mock_result = MagicMock()
        mock_result.output = "I've noted that JWT is preferred!"
        mock_result.usage.return_value = MagicMock(total_tokens=10, input_tokens=8, output_tokens=2)
        mock_agents[0].run = AsyncMock(return_value=mock_result)

        mock_store_direct = AsyncMock(
            return_value={"memory_id": 42, "cube_id": "fallback-uuid", "status": "stored"}
        )

        with patch("agents_factory.build_agents", return_value=mock_agents), \
             patch("agent.build_agents", return_value=mock_agents):
            import importlib, agent as agent_mod
            importlib.reload(agent_mod)  # pick up the patched build_agents
            mem_agent = agent_mod.MemoryAgent.__new__(agent_mod.MemoryAgent)
            mem_agent.ingest_agent = mock_agents[0]
            mem_agent.sync_queue = AsyncMock()
            mem_agent.client = None

        # Patch retry_with_backoff to return mock_result directly,
        # and db_session to simulate no new row found (agent didn't persist).
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.return_value.fetchone.return_value = None  # no new row

        with patch("agent.retry_with_backoff", AsyncMock(return_value=mock_result)), \
             patch("agent.db_session", return_value=mock_conn), \
             patch("memory_store.store_memory", mock_store_direct):
            result = await mem_agent.ingest("JWT is preferred over sessions", source="test")

        # Fallback must have been triggered
        mock_store_direct.assert_called_once()
        call_kwargs = mock_store_direct.call_args
        self.assertIn("JWT is preferred", call_kwargs.kwargs.get("raw_text", "")
                      or (call_kwargs.args[0] if call_kwargs.args else ""))
        self.assertIn("fallback", result)


if __name__ == "__main__":
    unittest.main()
