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
        # Disable REST auth for basic API tests (no keys, no auth check)
        os.environ.pop("AOM_API_KEYS", None)
        os.environ["AOM_REST_NO_AUTH"] = "true"
        init_db()
        self.agent = MockAgent()
        return build_http(self.agent, watch_path="./test_inbox")

    def tearDown(self):
        super().tearDown()
        os.environ.pop("AOM_REST_NO_AUTH", None)
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
        # origin_platform defaults to "rest-api" when no keys are configured
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

class TestRestAuth(AioHTTPTestCase):
    """Tests for REST API Bearer token authentication (Sprint 4).

    Uses AOM_API_KEYS=test-agent:validkey so that the middleware has keys to enforce.
    """

    async def get_application(self):
        db_file = "test_memory_auth.db"
        if os.path.exists(db_file):
            os.remove(db_file)
        os.environ["MEMORY_DB"] = db_file
        os.environ["AOM_API_KEYS"] = "test-agent:validkey"
        os.environ.pop("AOM_REST_NO_AUTH", None)
        init_db()
        self.agent = MockAgent()
        # build_http() reads env vars lazily at call time — no module reload needed
        return build_http(self.agent, watch_path="./test_inbox")

    def tearDown(self):
        super().tearDown()
        os.environ.pop("AOM_API_KEYS", None)
        for f in ("test_memory_auth.db",):
            if os.path.exists(f):
                os.remove(f)

    @unittest_run_loop
    async def test_ingest_requires_auth(self):
        """POST /ingest without a token → 401."""
        resp = await self.client.request("POST", "/ingest", json={"text": "hello"})
        self.assertEqual(resp.status, 401)

    @unittest_run_loop
    async def test_ingest_invalid_token(self):
        """POST /ingest with an unrecognised token → 403."""
        resp = await self.client.request(
            "POST", "/ingest", json={"text": "hello"},
            headers={"Authorization": "Bearer wrongkey"}
        )
        self.assertEqual(resp.status, 403)

    @unittest_run_loop
    async def test_ingest_valid_token(self):
        """POST /ingest with a valid token → 200 and caller_id embedded."""
        resp = await self.client.request(
            "POST", "/ingest", json={"text": "hello"},
            headers={"Authorization": "Bearer validkey"}
        )
        self.assertEqual(resp.status, 200)
        # Confirm ingest was called with the authenticated caller identity
        self.agent.ingest.assert_called_once_with(
            "hello", source="api", origin_platform="rest-api:test-agent"
        )

    @unittest_run_loop
    async def test_status_bypasses_auth(self):
        """GET /status should succeed without any token."""
        resp = await self.client.request("GET", "/status")
        self.assertEqual(resp.status, 200)


if __name__ == "__main__":
    unittest.main()
