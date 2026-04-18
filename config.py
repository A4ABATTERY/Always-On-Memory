"""
Configuration Module — Handles site-wide settings and environment variables.
"""

import asyncio
import importlib.util
import os
from pathlib import Path
from typing import Optional

# Global shutdown coordination — lazily created inside the running event loop
_shutdown_event: Optional[asyncio.Event] = None

def get_shutdown_event() -> asyncio.Event:
    """Return the process-wide shutdown event, creating it on first call.

    Must be called from within a running asyncio event loop (i.e. inside an
    async function or after asyncio.new_event_loop() has been set).  This
    avoids the DeprecationWarning / RuntimeError raised by Python 3.10-3.12
    when asyncio.Event() is instantiated at module-import time.
    """
    global _shutdown_event
    if _shutdown_event is None:
        _shutdown_event = asyncio.Event()
    return _shutdown_event

HAS_SQLITE_VEC: bool = importlib.util.find_spec("sqlite_vec") is not None

def _load_dotenv() -> None:
    """Load .env file from the script's directory (no external dependency)."""
    env_path = Path(__file__).parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # Handle quoted values first (before any comment stripping)
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        else:
            # Only strip trailing inline comments from unquoted values
            if " #" in value:
                value = value.split(" #")[0].strip()
        
        if key and key not in os.environ:  # don't override existing env vars
            os.environ[key] = value

_load_dotenv()

# ─── Config ────────────────────────────────────────────────────

def get_db_path() -> str:
    """Returns the current database path, allowing for dynamic override in tests."""
    return os.getenv("MEMORY_DB", "memory.db")

MODEL: str = os.getenv("MODEL", "gemini-3.1-flash-lite-preview")
SMART_MODEL: str = os.getenv("SMART_MODEL", "gemini-3-flash-preview")
RATE_LIMIT: int = int(os.getenv("RATE_LIMIT", "15"))
WATCH_DIRS: str = os.getenv("WATCH_DIRS", "")  # comma-separated folder paths
IGNORE_DIRS: str = os.getenv("IGNORE_DIRS", "")  # comma-separated extra dirs to skip
INBOX_DIR: str = os.getenv("INBOX_DIR", "inbox")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-2-preview")
SKILLS_DIR: str = os.getenv("SKILLS_DIR", ".agents/skills")
DEBOUNCE_INTERVAL: int = int(os.getenv("DEBOUNCE_INTERVAL", "10"))  # seconds to wait after last change before indexing
SCAN_INTERVAL: int = int(os.getenv("SCAN_INTERVAL", "5"))          # seconds between checking for modifications

# AutoDream configuration
IDLE_THRESHOLD_MINUTES: int = int(os.getenv("IDLE_THRESHOLD_MINUTES", "30"))
AUTODREAM_CHECK_INTERVAL: int = int(os.getenv("AUTODREAM_CHECK_INTERVAL", "300"))  # 5 min

# Verification loop — how often to scan WATCH_DIRS for missing/incomplete indexed files
VERIFY_INTERVAL_HOURS: int = int(os.getenv("VERIFY_INTERVAL_HOURS", "24"))
CONSOLIDATION_QUALITY_THRESHOLD: float = float(os.getenv("CONSOLIDATION_QUALITY_THRESHOLD", "0.85"))
DRIFT_THRESHOLD: float = float(os.getenv("DRIFT_THRESHOLD", "0.18"))
# Promotion threshold — distinct from DRIFT_THRESHOLD. A WorkDir file whose drift
# score exceeds this value is promoted to the Ingest Agent for full semantic ingestion.
# Set higher than DRIFT_THRESHOLD to avoid over-ingesting minor code edits.
PROMOTION_THRESHOLD: float = float(os.getenv("PROMOTION_THRESHOLD", "0.35"))

# MCP Server
MCP_PORT: int = int(os.getenv("MCP_PORT", "8765"))
MCP_HOST: str = os.getenv("MCP_HOST", "0.0.0.0")
AOM_API_KEYS: str = os.getenv("AOM_API_KEYS", "")  # format: "name1:key1,name2:key2"
# Set AOM_MCP_NO_AUTH=true ONLY for localhost-only deployments where auth is unnecessary.
# If unset (default), starting the MCP server without AOM_API_KEYS raises RuntimeError.
AOM_MCP_NO_AUTH: str = os.getenv("AOM_MCP_NO_AUTH", "")
# Set AOM_REST_NO_AUTH=true for localhost-only dev. When true, REST API requires no token.
# When false (default), Bearer token matching AOM_API_KEYS is required.
REST_NO_AUTH: bool = os.getenv("AOM_REST_NO_AUTH", "false").lower() in ("1", "true", "yes")


# Supported file types for multimodal ingestion (inbox watcher)
TEXT_EXTENSIONS: set[str] = {".txt", ".md", ".json", ".csv", ".log", ".xml", ".yaml", ".yml"}
MEDIA_EXTENSIONS: dict[str, str] = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp", ".bmp": "image/bmp",
    ".svg": "image/svg+xml",
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
    ".flac": "audio/flac", ".m4a": "audio/mp4", ".aac": "audio/aac",
    ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
    ".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
    ".pdf": "application/pdf",
}
ALL_SUPPORTED: set[str] = TEXT_EXTENSIONS | set(MEDIA_EXTENSIONS.keys())

# File types for Librarian (vector indexer)
CODE_EXTENSIONS: set[str] = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".go", ".rs", ".java", ".kt",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".sh",
    ".bash", ".zsh", ".sql", ".r", ".scala", ".dart",
    ".md", ".txt", ".rst", ".json", ".yaml", ".yml", ".toml", ".cfg",
    ".ini", ".env", ".dockerfile", ".makefile",
}

# Binary / machine-code extensions to always skip
BINARY_EXTENSIONS: set[str] = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".dylib", ".bin", ".exe",
    ".o", ".obj", ".a", ".lib", ".class", ".jar", ".war", ".ear",
    ".wasm", ".whl", ".egg", ".zip", ".tar", ".gz", ".bz2", ".xz",
    ".7z", ".rar", ".iso", ".dmg", ".deb", ".rpm",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg", ".webp",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv", ".flac",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    ".sqlite", ".db", ".sqlite3", ".lock",
    ".min.js", ".min.css", ".map",
}

# Directories to always skip during indexing
SKIP_DIRS: set[str] = {
    "__pycache__", "node_modules", ".git", ".hg", ".svn",
    "venv", ".venv", "env", ".env", ".tox", ".mypy_cache",
    ".pytest_cache", "dist", "build", ".eggs", "*.egg-info",
    ".next", ".nuxt", "coverage", ".coverage",
    "vendor", "target",  # Go/Rust build dirs
}
