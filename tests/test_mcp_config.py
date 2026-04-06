import os, unittest, unittest.mock
import importlib

class TestMCPConfig(unittest.TestCase):
    def test_mcp_port_default(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            # Remove vars if present
            os.environ.pop("MCP_PORT", None)
            os.environ.pop("MCP_HOST", None)
            import config
            importlib.reload(config)
            self.assertEqual(config.MCP_PORT, 8765)
            self.assertEqual(config.MCP_HOST, "0.0.0.0")

    def test_mcp_port_from_env(self):
        with unittest.mock.patch.dict(os.environ, {"MCP_PORT": "9000"}):
            import config
            importlib.reload(config)
            self.assertEqual(config.MCP_PORT, 9000)
