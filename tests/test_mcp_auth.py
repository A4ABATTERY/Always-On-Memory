import os, unittest, unittest.mock
import importlib

class TestMCPAuth(unittest.IsolatedAsyncioTestCase):

    def test_load_api_keys_empty(self):
        with unittest.mock.patch.dict(os.environ, {"AOM_API_KEYS": ""}):
            import mcp_auth
            importlib.reload(mcp_auth)
            keys = mcp_auth.load_api_keys()
            self.assertEqual(keys, {})

    def test_load_api_keys_single(self):
        with unittest.mock.patch.dict(os.environ, {"AOM_API_KEYS": "claude-desktop:abc123"}):
            import mcp_auth
            importlib.reload(mcp_auth)
            keys = mcp_auth.load_api_keys()
            self.assertEqual(keys, {"abc123": "claude-desktop"})

    def test_load_api_keys_multiple(self):
        with unittest.mock.patch.dict(os.environ, {"AOM_API_KEYS": "agent1:key1,agent2:key2"}):
            import mcp_auth
            importlib.reload(mcp_auth)
            keys = mcp_auth.load_api_keys()
            self.assertEqual(keys, {"key1": "agent1", "key2": "agent2"})

    def test_validate_key_valid(self):
        with unittest.mock.patch.dict(os.environ, {"AOM_API_KEYS": "claude:mykey"}):
            import mcp_auth
            importlib.reload(mcp_auth)
            keys = mcp_auth.load_api_keys()
            result = mcp_auth.validate_key("mykey", keys)
            self.assertEqual(result, "claude")

    def test_validate_key_invalid(self):
        with unittest.mock.patch.dict(os.environ, {"AOM_API_KEYS": "claude:mykey"}):
            import mcp_auth
            importlib.reload(mcp_auth)
            keys = mcp_auth.load_api_keys()
            result = mcp_auth.validate_key("wrongkey", keys)
            self.assertIsNone(result)

    def test_load_keys_ignores_malformed_entries(self):
        # Entry without colon separator is silently skipped
        with unittest.mock.patch.dict(os.environ, {"AOM_API_KEYS": "good:key1,badentry,also:key2"}):
            import mcp_auth
            importlib.reload(mcp_auth)
            keys = mcp_auth.load_api_keys()
            self.assertIn("key1", keys)
            self.assertIn("key2", keys)
            self.assertNotIn("badentry", keys.values())
