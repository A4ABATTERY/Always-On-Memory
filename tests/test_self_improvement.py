"""
Smoke tests for the Self-Improvement Agent.

Verifies that:
1. write_skill_file is registered as a tool on the self-improvement agent.
2. write_skill_file enforces a valid filename and creates a SKILL.md file.
3. The agent has the expected tools wired (read_memory_partition, search_documents,
   read_document, write_skill_file).

These tests do NOT make real LLM calls. They validate the agent's tooling
contract so that if the wiring breaks, tests fail before a live run.
"""

import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


class TestSelfImprovementAgentSmoke(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.test_skills_dir = "test_skills_tmp"
        self.env_patcher = unittest.mock.patch.dict(
            os.environ,
            {"MEMORY_DB": ":memory:", "SKILLS_DIR": self.test_skills_dir},
        )
        self.env_patcher.start()
        Path(self.test_skills_dir).mkdir(exist_ok=True)

    def tearDown(self):
        self.env_patcher.stop()
        if Path(self.test_skills_dir).exists():
            shutil.rmtree(self.test_skills_dir)

    def _get_tool_names(self, agent) -> set:
        """Return the set of tool names registered on an agent (PydanticAI 1.77+)."""
        ts = getattr(agent, '_function_toolset', None)
        if ts is not None and hasattr(ts, 'tools'):
            return set(ts.tools.keys())
        return set()

    def test_self_improvement_agent_has_write_skill_tool(self):
        """write_skill_file must be registered on the self-improvement agent."""
        from agents_factory import build_agents
        (*_, self_improvement_agent, _sync_agent) = build_agents()

        tool_names = self._get_tool_names(self_improvement_agent)
        self.assertIn(
            "write_skill_file",
            tool_names,
            f"write_skill_file not found in self-improvement agent tools. Found: {tool_names}",
        )

    def test_self_improvement_agent_has_all_expected_tools(self):
        """Verify the full tool set: read_memory_partition, search_documents, read_document, write_skill_file."""
        from agents_factory import build_agents
        (*_, self_improvement_agent, _sync_agent) = build_agents()

        expected = {"read_memory_partition", "search_documents", "read_document", "write_skill_file"}
        actual = self._get_tool_names(self_improvement_agent)
        missing = expected - actual
        self.assertFalse(
            missing,
            f"Self-improvement agent is missing tools: {missing}. Found: {actual}",
        )

    def test_write_skill_file_creates_skill_md(self):
        """write_skill_file must persist a SKILL.md under SKILLS_DIR/<skill_name>/."""
        from librarian import write_skill_file

        result = write_skill_file(
            skill_name="test-retry-pattern",
            content="# Retry Pattern\n\nAlways use exponential backoff.\n",
        )

        self.assertEqual(result["status"], "saved")
        skill_path = Path(result["path"])
        self.assertTrue(skill_path.exists(), f"SKILL.md not created at {skill_path}")
        self.assertEqual(skill_path.name, "SKILL.md")
        content = skill_path.read_text()
        self.assertIn("Retry Pattern", content)

    def test_write_skill_file_sanitises_name(self):
        """Skill names with unsafe characters must be sanitised."""
        from librarian import write_skill_file

        result = write_skill_file(
            skill_name="../../etc/passwd",
            content="# Malicious\n",
        )
        # Sanitised name should only contain safe chars
        clean_name = result["skill_name"]
        self.assertNotIn("..", clean_name)
        self.assertNotIn("/", clean_name)
        self.assertNotIn("\\", clean_name)

    async def test_self_improvement_agent_wired_for_run(self):
        """Agent can be constructed and run is callable (no real LLM needed)."""
        from agents_factory import build_agents
        (*_, self_improvement_agent, _) = build_agents()
        self.assertTrue(callable(getattr(self_improvement_agent, "run", None)))
        self.assertTrue(callable(getattr(self_improvement_agent, "run_sync", None)))


if __name__ == "__main__":
    unittest.main()
