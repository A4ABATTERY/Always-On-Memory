"""
Tests for the Sprint 6 JSONL Audit Trail — audit.py, database.py, memory_store.py, server.py.

NOTE: SQLite `:memory:` creates a fresh empty DB per connection, so tests that need the
schema to persist across connections use a named temp file (same pattern as test_api.py).
"""

import asyncio
import glob
import json
import os
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock


def _make_temp_db():
    """Return path to a fresh temp SQLite file with the full schema initialised."""
    tf = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tf.close()
    os.environ["MEMORY_DB"] = tf.name
    from database import init_db
    init_db()
    return tf.name


class TestWriteAuditEvent(unittest.IsolatedAsyncioTestCase):
    """Unit tests for audit.write_audit_event()."""

    def setUp(self):
        self.db_path = _make_temp_db()

    def tearDown(self):
        os.environ.pop("MEMORY_DB", None)
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    async def test_created_event_written_to_db(self):
        """write_audit_event('created') inserts a row into memory_audit_log."""
        import audit
        with patch_audit_dir(""):
            await audit.write_audit_event(
                cube_id="test-cube-001",
                event_type="created",
                actor="inbox-watcher",
                after_text="Some raw text",
                metadata={"memory_id": 1},
            )

        from database import db_session
        with db_session() as db:
            row = db.execute(
                "SELECT * FROM memory_audit_log WHERE cube_id = ?",
                ("test-cube-001",),
            ).fetchone()

        self.assertIsNotNone(row, "Expected a row in memory_audit_log")
        self.assertEqual(row["event_type"], "created")
        self.assertEqual(row["actor"], "inbox-watcher")
        self.assertEqual(row["after_text"], "Some raw text")
        self.assertIsNone(row["before_text"])

    async def test_repaired_event_stores_before_and_after(self):
        """write_audit_event('repaired') preserves both before_text and after_text."""
        import audit
        with patch_audit_dir(""):
            await audit.write_audit_event(
                cube_id="test-cube-002",
                event_type="repaired",
                actor="sync-auditor",
                before_text="Old text",
                after_text="New text",
            )

        from database import db_session
        with db_session() as db:
            row = db.execute(
                "SELECT * FROM memory_audit_log WHERE cube_id = ?",
                ("test-cube-002",),
            ).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row["event_type"], "repaired")
        self.assertEqual(row["before_text"], "Old text")
        self.assertEqual(row["after_text"], "New text")

    async def test_jsonl_file_written(self):
        """write_audit_event writes one JSON line to the daily JSONL file."""
        import audit
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch_audit_dir(tmpdir):
                await audit.write_audit_event(
                    cube_id="test-cube-003",
                    event_type="created",
                    actor="test",
                    after_text="hello",
                )

            files = glob.glob(f"{tmpdir}/*.jsonl")
            self.assertEqual(len(files), 1, "Expected exactly one JSONL file")
            with open(files[0], encoding="utf-8") as f:
                line = json.loads(f.readline())

        self.assertEqual(line["cube_id"], "test-cube-003")
        self.assertEqual(line["event_type"], "created")

    async def test_jsonl_skipped_when_dir_empty(self):
        """When AOM_AUDIT_LOG_DIR='', no JSONL file is written but no error is raised."""
        import audit
        with patch_audit_dir(""):
            # Should complete without raising.
            await audit.write_audit_event(
                cube_id="test-cube-004",
                event_type="created",
                actor="test",
                after_text="no file",
            )


class TestAuditEndpoint(unittest.IsolatedAsyncioTestCase):
    """Integration test for the GET /audit REST endpoint."""

    async def asyncSetUp(self):
        # Use file-based DB so schema persists across connections.
        self.db_path = _make_temp_db()
        os.environ["AOM_REST_NO_AUTH"] = "true"
        os.environ.pop("AOM_API_KEYS", None)

        # Seed one audit event directly into the DB.
        from database import db_session
        from datetime import datetime, timezone
        with db_session() as db:
            db.execute(
                """INSERT INTO memory_audit_log
                   (cube_id, event_type, actor, before_text, after_text, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                ("seed-cube", "created", "inbox-watcher", None, "hello world",
                 json.dumps({}), datetime.now(timezone.utc).isoformat()),
            )
            db.commit()

        from server import build_http
        mock_agent = MagicMock()
        mock_agent.query = AsyncMock(return_value="x")
        mock_agent.ingest = AsyncMock(return_value="x")
        mock_agent.consolidate = AsyncMock(return_value="x")
        mock_agent.deep_reconsolidate = AsyncMock(return_value="x")
        mock_agent.self_improve = AsyncMock(return_value="x")
        mock_agent.clear = AsyncMock(return_value={"status": "cleared"})

        from aiohttp.test_utils import TestClient, TestServer
        self.app = build_http(mock_agent, watch_path="./test_inbox")
        self.server = TestServer(self.app)
        self.client = TestClient(self.server)
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()
        os.environ.pop("AOM_REST_NO_AUTH", None)
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    async def test_audit_by_cube_id(self):
        resp = await self.client.get("/audit?cube_id=seed-cube")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertEqual(data["count"], 1)
        self.assertEqual(data["events"][0]["event_type"], "created")
        self.assertEqual(data["events"][0]["actor"], "inbox-watcher")

    async def test_audit_missing_params_returns_400(self):
        resp = await self.client.get("/audit")
        self.assertEqual(resp.status, 400)

    async def test_audit_cube_id_not_found_returns_empty(self):
        resp = await self.client.get("/audit?cube_id=nonexistent")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertEqual(data["count"], 0)

    async def test_audit_since_returns_events(self):
        resp = await self.client.get("/audit?since=2000-01-01")
        self.assertEqual(resp.status, 200)
        data = await resp.json()
        self.assertGreaterEqual(data["count"], 1)


# ─── Helpers ─────────────────────────────────────────────────────────────────

from contextlib import contextmanager
from unittest.mock import patch


@contextmanager
def patch_audit_dir(path: str):
    """Temporarily override AOM_AUDIT_LOG_DIR in the audit module."""
    with patch.dict(os.environ, {"AOM_AUDIT_LOG_DIR": path}):
        import importlib, config as _cfg, audit as _audit
        importlib.reload(_cfg)
        importlib.reload(_audit)
        yield


if __name__ == "__main__":
    unittest.main()
