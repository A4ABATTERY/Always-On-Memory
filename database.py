"""
Database Module — Manages SQLite connections and schema initialization.
"""

import contextlib
import logging
import sqlite3
import os
from typing import Generator, Any

try:
    import sqlite_vec
    HAS_SQLITE_VEC = True
except ImportError:
    HAS_SQLITE_VEC = False

from config import DB_PATH

log = logging.getLogger("memory-agent.database")

def get_db() -> sqlite3.Connection:
    """Get a fresh database connection, re-evaluating the path from env."""
    db = sqlite3.connect(DB_PATH) # Removed timeout parameter
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
                source TEXT NOT NULL DEFAULT '',
                raw_text TEXT NOT NULL,
                summary TEXT NOT NULL,
                entities TEXT NOT NULL DEFAULT '[]',
                topics TEXT NOT NULL DEFAULT '[]',
                connections TEXT NOT NULL DEFAULT '[]',
                importance REAL NOT NULL DEFAULT 0.5,
                created_at TEXT NOT NULL,
                consolidated INTEGER NOT NULL DEFAULT 0,
                sector TEXT NOT NULL DEFAULT 'semantic',
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

        # Create vec0 virtual table for vector search (if sqlite-vec loaded)
        if HAS_SQLITE_VEC:
            try:
                # Check if vec_documents already exists
                db.execute("SELECT 1 FROM vec_documents LIMIT 1").fetchone()
            except sqlite3.OperationalError:
                try:
                    db.execute("""
                        CREATE VIRTUAL TABLE vec_documents USING vec0(
                            document_id INTEGER PRIMARY KEY,
                            embedding float[3072]
                        )
                    """)
                except Exception as e:
                    log.warning(f"Could not create vec_documents: {e}")

        # Migrations for existing DB
        for col_sql in [
            "ALTER TABLE memories ADD COLUMN sector TEXT NOT NULL DEFAULT 'semantic'",
            "ALTER TABLE memories ADD COLUMN valid_to TEXT DEFAULT NULL",
        ]:
            try:
                db.execute(col_sql)
            except sqlite3.OperationalError:
                pass
