import asyncio
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# Use in-memory DB so no file is created
os.environ.setdefault("MEMORY_DB", ":memory:")


class TestCoreMCPTools(unittest.IsolatedAsyncioTestCase):
    """Tests for the four core semantic MCP tools."""

    def setUp(self):
        self.env_patcher = patch.dict(os.environ, {"MEMORY_DB": ":memory:"})
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()
        # Reset shared module-level state so tests do not bleed into each other
        import mcp_server
        mcp_server._agent = None

    async def test_remember_calls_agent_ingest(self):
        # FastMCP 3.x decorator_mode="function": @mcp.tool() returns the original
        # coroutine unchanged. Call it directly — no .fn attribute.
        mock_agent = MagicMock()
        mock_agent.ingest = AsyncMock(return_value="Stored as MemCube #1")

        import mcp_server
        mcp_server._agent = mock_agent

        from mcp_server import remember
        result = await remember("JWT auth preferred over sessions", source="test")

        mock_agent.ingest.assert_called_once_with(
            "JWT auth preferred over sessions", source="test", origin_platform="unknown"
        )
        self.assertEqual(result, "Stored as MemCube #1")

    async def test_recall_calls_agent_query(self):
        mock_agent = MagicMock()
        mock_agent.query = AsyncMock(return_value="JWT is preferred.")

        import mcp_server
        mcp_server._agent = mock_agent

        from mcp_server import recall
        result = await recall("What auth approach do we use?")

        mock_agent.query.assert_called_once_with("What auth approach do we use?")
        self.assertEqual(result, "JWT is preferred.")

    async def test_status_calls_get_memory_stats(self):
        fake_stats = {"total_memories": 5, "unconsolidated": 2, "consolidations": 1}

        import mcp_server
        mcp_server._agent = MagicMock()

        with patch("memory_store.get_memory_stats", return_value=fake_stats):
            from mcp_server import status
            result = await status()

        self.assertEqual(result, fake_stats)
        self.assertEqual(result["total_memories"], 5)

    async def test_forget_calls_delete_memory(self):
        fake_result = {"status": "deleted", "memory_id": 7}

        import mcp_server
        mcp_server._agent = MagicMock()

        with patch("memory_store.delete_memory", return_value=fake_result):
            from mcp_server import forget
            result = await forget(7)

        self.assertEqual(result["status"], "deleted")
        self.assertEqual(result["memory_id"], 7)

    async def test_forget_invalid_id_returns_not_found(self):
        fake_result = {"status": "not_found", "memory_id": 999}

        import mcp_server
        mcp_server._agent = MagicMock()

        with patch("memory_store.delete_memory", return_value=fake_result):
            from mcp_server import forget
            result = await forget(999)

        self.assertEqual(result["status"], "not_found")


class TestPowerMCPTools(unittest.IsolatedAsyncioTestCase):
    """Tests for the nine power MCP tools."""

    def setUp(self):
        self.env_patcher = patch.dict(os.environ, {"MEMORY_DB": ":memory:"})
        self.env_patcher.start()
        import mcp_server
        mcp_server._agent = MagicMock()

    def tearDown(self):
        self.env_patcher.stop()
        # Reset shared module-level state so tests do not bleed into each other
        import mcp_server
        mcp_server._agent = None

    async def test_list_memories_calls_read_all(self):
        fake = {"memories": [{"id": 1, "summary": "test"}], "count": 1}
        with patch("memory_store.read_all_memories", return_value=fake) as mock_fn:
            from mcp_server import list_memories
            result = await list_memories(limit=25)
            mock_fn.assert_called_once_with(limit=25)
            self.assertEqual(result["count"], 1)

    async def test_search_documents_calls_librarian(self):
        fake = {"results": [{"path": "agent.py", "snippet": "def ingest"}], "count": 1}
        with patch("librarian.search_documents", new=AsyncMock(return_value=fake)) as mock_fn:
            from mcp_server import search_documents
            result = await search_documents("ingest", limit=3)
            mock_fn.assert_called_once_with("ingest", k=3)
            self.assertEqual(result["count"], 1)

    async def test_consolidate_calls_agent(self):
        import mcp_server
        mcp_server._agent.consolidate = AsyncMock(return_value="Consolidated 5 memories.")
        from mcp_server import consolidate
        result = await consolidate()
        mcp_server._agent.consolidate.assert_called_once()
        self.assertIn("5", result)

    async def test_export_memories_no_filter(self):
        fake = {"cubes": [], "count": 0}
        with patch("memory_store.export_cubes", return_value=fake) as mock_fn:
            from mcp_server import export_memories
            result = await export_memories()
            mock_fn.assert_called_once_with(memory_ids=None)
            self.assertEqual(result, fake)  # verify return value is passed through

    async def test_export_memories_with_ids(self):
        fake = {"cubes": [{"cube_id": "abc"}], "count": 1}
        with patch("memory_store.export_cubes", return_value=fake) as mock_fn:
            from mcp_server import export_memories
            result = await export_memories(memory_ids=[1, 2])
            mock_fn.assert_called_once_with(memory_ids=[1, 2])

    async def test_import_memories_calls_import_cubes(self):
        fake = {"status": "success", "imported": 2}
        with patch("memory_store.import_cubes", new=AsyncMock(return_value=fake)):
            from mcp_server import import_memories
            result = await import_memories([{"cube_id": "x"}, {"cube_id": "y"}])
            self.assertEqual(result["imported"], 2)

    async def test_list_links_calls_get_all_links(self):
        fake = {"links": [], "count": 0}
        with patch("memory_store.get_all_links", return_value=fake):
            from mcp_server import list_links
            result = await list_links()
            self.assertEqual(result["count"], 0)

    async def test_reinforce_calls_reinforce_memory(self):
        fake = {"status": "reinforced", "memory_id": 3, "new_importance": 0.7}
        with patch("memory_store.reinforce_memory", return_value=fake):
            from mcp_server import reinforce
            result = await reinforce(3)
            self.assertEqual(result["status"], "reinforced")

    async def test_self_improve_calls_agent(self):
        import mcp_server
        mcp_server._agent.self_improve = AsyncMock(return_value="Wrote 2 skill files.")
        from mcp_server import self_improve
        result = await self_improve()
        self.assertIn("skill", result.lower())


class TestMCPServerStartup(unittest.IsolatedAsyncioTestCase):

    def test_run_mcp_server_is_coroutine(self):
        """run_mcp_server must be an async function so main_async() can task it."""
        import inspect
        import mcp_server
        self.assertTrue(
            hasattr(mcp_server, "run_mcp_server") and inspect.iscoroutinefunction(mcp_server.run_mcp_server),
            "run_mcp_server must be async",
        )

    async def test_mcp_has_expected_tool_count(self):
        """FastMCP instance must expose exactly 13 tools via the public list_tools() API.
        FastMCP 3.x decorator_mode='function' keeps the original coroutine as the tool
        function; tools are enumerable via await mcp.list_tools() without a running server.
        """
        import mcp_server
        tools = await mcp_server.mcp.list_tools()
        tool_names = [t.name for t in tools]
        self.assertEqual(
            len(tools), 13,
            f"Expected 13 tools, got {len(tools)}: {tool_names}",
        )
