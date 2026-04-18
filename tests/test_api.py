"""
API Integration tests for Always-On-Memory Agent.
"""

import os
import unittest
from unittest.mock import AsyncMock, MagicMock
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

# Mock agent for testing routes in isolation
class MockAgent:
    def __init__(self):
        self.query = AsyncMock(return_value="Mock Answer")
        self.ingest = AsyncMock(return_value="Mock Ingested")
        self.consolidate = AsyncMock(return_value="Mock Consolidated")
        self.deep_reconsolidate = AsyncMock(return_value="Mock Deep")
        self.self_improve = AsyncMock(return_value="Mock Improved")
        self.clear = AsyncMock(return_value={"status": "cleared"})

from server import build_http
from database import init_db

class TestAPI(AioHTTPTestCase):

    async def get_application(self):
        # Ensure test DB is ready
        db_file = "test_memory_api.db"
        if os.path.exists(db_file):
            os.remove(db_file)
        os.environ["MEMORY_DB"] = db_file
        init_db()
        self.agent = MockAgent()
        return build_http(self.agent, watch_path="./test_inbox")

    def tearDown(self):
        super().tearDown()
        if os.path.exists("test_memory_api.db"):
            os.remove("test_memory_api.db")

    @unittest_run_loop
    async def test_query_route(self):
        resp = await self.client.request("GET", "/query?q=hello")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertEqual(data["answer"], "Mock Answer")
        self.agent.query.assert_called_once_with("hello")

    @unittest_run_loop
    async def test_ingest_route(self):
        payload = {"text": "some info", "source": "web"}
        resp = await self.client.request("POST", "/ingest", json=payload)
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertEqual(data["status"], "ingested")
        self.agent.ingest.assert_called_once_with("some info", source="web", origin_platform="rest-api")

    @unittest_run_loop
    async def test_status_route(self):
        resp = await self.client.request("GET", "/status")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertIn("total_memories", data)

    @unittest_run_loop
    async def test_export_import(self):
        # 1. Export (should be empty but 200)
        resp = await self.client.request("GET", "/export_cubes")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertEqual(len(data["cubes"]), 0)

        # 2. Import a mock cube
        cube = {
            "cube_id": "test-item-1",
            "sector": "semantic",
            "raw_text": "hello portable world",
            "summary": "Imported memory",
            "entities": ["portable"],
            "topics": ["test"],
            "importance_score": 0.9,
            "created_at": "2024-01-01T00:00:00Z"
        }
        resp = await self.client.request("POST", "/import_cubes", json=[cube])
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertEqual(data["imported"], 1)

if __name__ == "__main__":
    unittest.main()
