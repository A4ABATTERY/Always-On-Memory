"""
Tests for Visibility & Monitoring API endpoints (V3.3).
Verifies /status and /links endpoints.
"""

import unittest
import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from server import build_http
from database import init_db, db_session
from memory_store import store_memory

class TestMonitoringAPI(AioHTTPTestCase):
    
    async def get_application(self):
        self.db_path = "test_api_monitoring.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.environ["MEMORY_DB"] = self.db_path
        init_db()
        
        self.agent = MagicMock()
        # Mocking async methods
        self.agent.query = AsyncMock(return_value="Answer")
        self.agent.ingest = AsyncMock(return_value="Ingested")
        
        return build_http(self.agent)

    def tearDown(self):
        super().tearDown()
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    @unittest_run_loop
    async def test_status_and_links_endpoints(self):
        # 1. Seed some links
        await store_memory(
            raw_text="Logic A", summary="Summary A",
            entities=[], topics=[], importance_score=0.8,
            connections=[{"type": "file_link", "path": "/path/to/a.py", "status": "active"}]
        )
        await store_memory(
            raw_text="Logic B", summary="Summary B",
            entities=[], topics=[], importance_score=0.8,
            connections=[{"type": "file_link", "path": "/path/to/b.py", "status": "historical_trace"}]
        )
        
        # 2. Check /status
        async with self.client.get("/status") as resp:
            self.assertEqual(resp.status, 200)
            data = await resp.json()
            self.assertIn("structural_links", data)
            self.assertEqual(data["structural_links"]["active"], 1)
            self.assertEqual(data["structural_links"]["historical_trace"], 1)
            self.assertEqual(data["structural_links"]["total"], 2)
            
        # 3. Check /links
        async with self.client.get("/links") as resp:
            self.assertEqual(resp.status, 200)
            data = await resp.json()
            self.assertIn("links", data)
            self.assertEqual(len(data["links"]), 2)
            # Verify one link details
            link_a = next(l for l in data["links"] if l["path"] == "/path/to/a.py")
            self.assertEqual(link_a["status"], "active")
            self.assertEqual(link_a["memory_summary"], "Summary A")

if __name__ == "__main__":
    unittest.main()
