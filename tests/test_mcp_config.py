import os, pathlib, unittest, unittest.mock
import importlib

class TestMCPConfig(unittest.TestCase):
    def test_mcp_port_default(self):
        # Pre-import so _load_dotenv() runs now (before we manipulate os.environ).
        # Without this, a first-time import inside the patch.dict context would call
        # _load_dotenv() AFTER the pop, re-adding MCP_HOST from .env into os.environ.
        import config
        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MCP_PORT", None)
            os.environ.pop("MCP_HOST", None)
            # Patch Path.exists so _load_dotenv sees no .env file during reload.
            # patch.object(config, "_load_dotenv") doesn't work because reload()
            # re-executes the source, re-defining the function before calling it.
            with unittest.mock.patch.object(pathlib.Path, "exists", return_value=False):
                importlib.reload(config)
            self.assertEqual(config.MCP_PORT, 8765)
            self.assertEqual(config.MCP_HOST, "0.0.0.0")

    def test_mcp_port_from_env(self):
        with unittest.mock.patch.dict(os.environ, {"MCP_PORT": "9000"}):
            import config
            importlib.reload(config)
            self.assertEqual(config.MCP_PORT, 9000)
