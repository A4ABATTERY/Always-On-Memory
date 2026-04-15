"""
Database Module — Manages SQLite connections and schema initialization.
"""

import contextlib
import logging
import sqlite3
from typing import Generator

try:
    import sqlite_vec
    HAS_SQLITE_VEC = True
except ImportError:
    HAS_SQLITE_VEC = False

from config import get_db_path

log = logging.getLogger("memory-agent.database")

def get_db() -> sqlite3.Connection:
    """Get a fresh database connection, re-evaluating the path from env."""
    db = sqlite3.connect(get_db_path())
    db.row_factory = sqlite3.Row
    try:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        db.execute("PRAGMA busy_timeout=5000")
    except Exception as e:
        log.debug(f"Pragma application failed: {e}")
        
    if HAS_SQLITE_VEC:
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
    return db

@contextlib.contextmanager
def db_session() -> Generator[sqlite3.Connection, None, None]:
    """Context manager that ensures the database connection is closed."""
    db = get_db()
    try:
        yield db
    finally:
        db.close()

def init_db() -> None:
    """Initialize database schema and run migrations."""
    with db_session() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cube_id TEXT NOT NULL UNIQUE,
                sector TEXT NOT NULL DEFAULT 'semantic',
                source TEXT NOT NULL DEFAULT '',
                origin_platform TEXT NOT NULL DEFAULT 'aom-local',
                raw_text TEXT NOT NULL,
                summary TEXT NOT NULL,
                entities TEXT NOT NULL DEFAULT '[]',
                topics TEXT NOT NULL DEFAULT '[]',
                connections TEXT NOT NULL DEFAULT '[]',
                metadata TEXT NOT NULL DEFAULT '{}',
                importance_score REAL NOT NULL DEFAULT 0.5,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed TEXT DEFAULT NULL,
                created_at TEXT NOT NULL,
                consolidated INTEGER NOT NULL DEFAULT 0,
                valid_to TEXT DEFAULT NULL
            );
            CREATE TABLE IF NOT EXISTS consolidations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_ids TEXT NOT NULL,
                summary TEXT NOT NULL,
                insight TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS processed_files (
                path TEXT PRIMARY KEY,
                processed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_index INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL,
                UNIQUE(path, chunk_index)
            );
        """)

        # Additive migrations for processed_files
        try:
            db.execute("ALTER TABLE processed_files ADD COLUMN content_hash TEXT DEFAULT NULL")
        except sqlite3.OperationalError:
            pass  # Column already exists

        try:
            db.execute("ALTER TABLE processed_files ADD COLUMN prev_hash TEXT DEFAULT NULL")
        except sqlite3.OperationalError:
            pass  # Column already exists

        # Create vec0 virtual tables for vector search (if sqlite-vec loaded)
        if HAS_SQLITE_VEC:
            try:
                # Check if vec_documents already exists
                db.execute("SELECT 1 FROM vec_documents LIMIT 1").fetchone()
            except sqlite3.OperationalError:
                try:
                    db.execute("""
                        CREATE VIRTUAL TABLE vec_documents USING vec0(
                            document_id INTEGER PRIMARY KEY,
                            embedding int8[3072]
                        )
                    """)
                except Exception as e:
                    log.warning(f"Could not create vec_documents: {e}")

            try:
                # Check if vec_memories already exists
                db.execute("SELECT 1 FROM vec_memories LIMIT 1").fetchone()
            except sqlite3.OperationalError:
                try:
                    db.execute("""
                        CREATE VIRTUAL TABLE vec_memories USING vec0(
                            memory_id INTEGER PRIMARY KEY,
                            embedding int8[3072]
                        )
                    """)
                except Exception as e:
                    log.warning(f"Could not create vec_memories: {e}")
