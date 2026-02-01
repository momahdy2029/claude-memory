"""Database service using SQLite with FAISS vector indexing.

Uses FAISS for O(log n) similarity search when available,
falls back to numpy-based O(n) search otherwise.

Features:
- Connection pooling for SQLite (thread-safe connections)
- Retry logic with exponential backoff for transient failures
- Query timeout handling
- Comprehensive error handling with logging
"""
import os
import json
import sqlite3
import numpy as np
import logging
import time
import threading
from queue import Queue, Empty
from functools import wraps
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple, Callable
from pathlib import Path
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DB_PATH = os.getenv("DATABASE_PATH", str(Path(__file__).parent.parent / "memories.db"))
USE_VECTOR_INDEX = os.getenv("USE_VECTOR_INDEX", "true").lower() == "true"

# Connection pool settings
DB_POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "5"))
DB_TIMEOUT = float(os.getenv("DB_TIMEOUT", "30.0"))  # Query timeout in seconds
DB_MAX_RETRIES = int(os.getenv("DB_MAX_RETRIES", "3"))
DB_RETRY_BASE_DELAY = float(os.getenv("DB_RETRY_BASE_DELAY", "0.1"))  # Base delay for exponential backoff


# Custom exceptions for structured error handling
class DatabaseError(Exception):
    """Base exception for database errors."""
    def __init__(self, message: str, error_code: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.error_code = error_code
        self.original_error = original_error


class ConnectionPoolError(DatabaseError):
    """Error related to connection pool."""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message, "DB_POOL_ERROR", original_error)


class QueryTimeoutError(DatabaseError):
    """Query execution timeout."""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message, "DB_TIMEOUT", original_error)


class RetryExhaustedError(DatabaseError):
    """All retry attempts failed."""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message, "DB_RETRY_EXHAUSTED", original_error)


class MigrationError(DatabaseError):
    """Database migration failed."""
    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message, "DB_MIGRATION_ERROR", original_error)


class SQLiteConnectionPool:
    """Thread-safe connection pool for SQLite.

    SQLite has limited connection pooling needs compared to client-server DBs,
    but this provides:
    - Thread-safe connection management
    - Connection reuse to avoid repeated file opens
    - Graceful connection lifecycle management
    """

    def __init__(self, db_path: str, pool_size: int = 5, timeout: float = 30.0):
        self.db_path = db_path
        self.pool_size = pool_size
        self.timeout = timeout
        self._pool: Queue = Queue(maxsize=pool_size)
        self._lock = threading.Lock()
        self._created_connections = 0
        self._active_connections = 0

    def _create_connection(self) -> sqlite3.Connection:
        """Create a new SQLite connection with optimal settings."""
        conn = sqlite3.connect(
            self.db_path,
            timeout=self.timeout,
            check_same_thread=False,
            isolation_level=None  # Autocommit mode for better concurrency
        )
        conn.row_factory = sqlite3.Row
        # Enable WAL mode for better concurrent read/write performance
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        conn.execute("PRAGMA busy_timeout=30000")  # 30 second busy timeout
        return conn

    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool or create a new one."""
        try:
            # Try to get from pool (non-blocking first)
            conn = self._pool.get_nowait()
            self._active_connections += 1
            return conn
        except Empty:
            pass

        # Create new connection if pool not full
        with self._lock:
            if self._created_connections < self.pool_size:
                conn = self._create_connection()
                self._created_connections += 1
                self._active_connections += 1
                logger.debug(f"Created new connection (total: {self._created_connections})")
                return conn

        # Pool is full, wait for available connection
        try:
            conn = self._pool.get(timeout=self.timeout)
            self._active_connections += 1
            return conn
        except Empty:
            raise ConnectionPoolError(
                f"Connection pool exhausted (size={self.pool_size}, timeout={self.timeout}s)"
            )

    def return_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool."""
        if conn is None:
            return

        self._active_connections -= 1

        try:
            # Check if connection is still valid
            conn.execute("SELECT 1")
            self._pool.put_nowait(conn)
        except (sqlite3.Error, sqlite3.ProgrammingError):
            # Connection is bad, close it
            try:
                conn.close()
            except Exception:
                pass
            with self._lock:
                self._created_connections -= 1
            logger.warning("Closed invalid connection from pool")

    def close_all(self):
        """Close all connections in the pool."""
        with self._lock:
            while not self._pool.empty():
                try:
                    conn = self._pool.get_nowait()
                    conn.close()
                except Empty:
                    break
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")
            self._created_connections = 0
            self._active_connections = 0
        logger.info("Connection pool closed")

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "pool_size": self.pool_size,
            "created_connections": self._created_connections,
            "active_connections": self._active_connections,
            "available_connections": self._pool.qsize(),
            "timeout": self.timeout
        }


def with_retry(
    max_retries: int = DB_MAX_RETRIES,
    base_delay: float = DB_RETRY_BASE_DELAY,
    retryable_errors: tuple = (sqlite3.OperationalError, sqlite3.DatabaseError)
):
    """Decorator for retry logic with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (will be multiplied exponentially)
        retryable_errors: Tuple of exception types that should trigger retry
    """
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except retryable_errors as e:
                    last_error = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {delay:.2f}s due to: {str(e)}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries} retries exhausted for {func.__name__}: {str(e)}"
                        )
            raise RetryExhaustedError(
                f"Operation {func.__name__} failed after {max_retries} retries",
                original_error=last_error
            )

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_errors as e:
                    last_error = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__} "
                            f"after {delay:.2f}s due to: {str(e)}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_retries} retries exhausted for {func.__name__}: {str(e)}"
                        )
            raise RetryExhaustedError(
                f"Operation {func.__name__} failed after {max_retries} retries",
                original_error=last_error
            )

        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


def normalize_path(path: str) -> str:
    """Normalize file paths to prevent duplicates from different separators.

    Converts all paths to forward slashes (Unix-style) for consistency.
    This prevents 'C:/foo' and 'C:\\foo' being treated as different projects.
    Also normalizes Windows drive letters to uppercase for case-insensitive matching.
    """
    if not path:
        return path
    # Convert to forward slashes and remove trailing slashes
    normalized = path.replace("\\", "/").rstrip("/")
    # Normalize Windows drive letter to uppercase (c: -> C:)
    if len(normalized) >= 2 and normalized[1] == ':':
        normalized = normalized[0].upper() + normalized[1:]
    return normalized


class DatabaseService:
    """Service for vector storage and retrieval using SQLite + FAISS.

    Features:
    - FAISS vector indexing for O(log n) similarity search
    - Automatic index building on startup
    - Incremental index updates on insert
    - Fallback to numpy-based search if FAISS unavailable
    - Connection pooling for thread-safe access
    - Retry logic with exponential backoff
    - Query timeout handling
    """

    def __init__(self):
        self.db_path = DB_PATH
        self.conn: Optional[sqlite3.Connection] = None
        self._connection_pool: Optional[SQLiteConnectionPool] = None

        # Vector indexes (lazy loaded)
        self._memories_index = None
        self._patterns_index = None
        self._timeline_index = None
        self._use_vector_index = USE_VECTOR_INDEX
        self._index_initialized = False

    @contextmanager
    def get_connection(self):
        """Context manager for getting a connection from the pool.

        Usage:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(...)

        Falls back to self.conn if pool not initialized.
        """
        if self._connection_pool:
            conn = self._connection_pool.get_connection()
            try:
                yield conn
            finally:
                self._connection_pool.return_connection(conn)
        else:
            # Fallback for backward compatibility
            yield self.conn

    async def connect(self):
        """Establish database connection and initialize connection pool."""
        try:
            # Initialize connection pool
            self._connection_pool = SQLiteConnectionPool(
                db_path=self.db_path,
                pool_size=DB_POOL_SIZE,
                timeout=DB_TIMEOUT
            )
            # Keep a primary connection for backward compatibility
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            # Enable WAL mode on primary connection too
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA busy_timeout=30000")
            logger.info(f"Database connected with pool size {DB_POOL_SIZE}")
        except sqlite3.Error as e:
            logger.error(f"Failed to connect to database: {e}")
            raise ConnectionPoolError(f"Failed to connect to database: {e}", original_error=e)

    async def disconnect(self):
        """Close database connection, connection pool, and save indexes."""
        # Save indexes
        if self._memories_index:
            try:
                self._memories_index.save()
            except Exception as e:
                logger.warning(f"Failed to save memories index: {e}")
        if self._patterns_index:
            try:
                self._patterns_index.save()
            except Exception as e:
                logger.warning(f"Failed to save patterns index: {e}")
        if self._timeline_index:
            try:
                self._timeline_index.save()
            except Exception as e:
                logger.warning(f"Failed to save timeline index: {e}")

        # Close connection pool
        if self._connection_pool:
            self._connection_pool.close_all()
            self._connection_pool = None

        # Close primary connection
        if self.conn:
            try:
                self.conn.close()
            except Exception as e:
                logger.warning(f"Error closing primary connection: {e}")
            self.conn = None

        logger.info("Database disconnected")

    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        if self._connection_pool:
            return self._connection_pool.get_stats()
        return {"pool_initialized": False}

    async def _init_vector_indexes(self):
        """Initialize vector indexes from database."""
        if self._index_initialized or not self._use_vector_index:
            return

        try:
            from services.vector_index import get_index

            # Initialize memories index
            self._memories_index = get_index("memories")
            if self._memories_index.size() == 0:
                await self._rebuild_memories_index()

            # Initialize patterns index
            self._patterns_index = get_index("patterns")
            if self._patterns_index.size() == 0:
                await self._rebuild_patterns_index()

            # Initialize timeline index
            self._timeline_index = get_index("timeline")
            if self._timeline_index.size() == 0:
                await self._rebuild_timeline_index()

            self._index_initialized = True
        except ImportError:
            # FAISS not available, will use numpy fallback
            self._use_vector_index = False

    async def _rebuild_memories_index(self):
        """Rebuild the memories vector index from database."""
        if not self._memories_index:
            return

        cursor = self.conn.cursor()
        cursor.execute("SELECT id, embedding FROM memories WHERE embedding IS NOT NULL")
        rows = cursor.fetchall()

        items = []
        for row in rows:
            embedding = self._deserialize_embedding(row["embedding"])
            if embedding:
                items.append((row["id"], embedding))

        if items:
            self._memories_index.rebuild(items)
            self._memories_index.save()

    async def _rebuild_patterns_index(self):
        """Rebuild the patterns vector index from database."""
        if not self._patterns_index:
            return

        cursor = self.conn.cursor()
        cursor.execute("SELECT id, embedding FROM patterns WHERE embedding IS NOT NULL")
        rows = cursor.fetchall()

        items = []
        for row in rows:
            embedding = self._deserialize_embedding(row["embedding"])
            if embedding:
                items.append((row["id"], embedding))

        if items:
            self._patterns_index.rebuild(items)
            self._patterns_index.save()

    async def _rebuild_timeline_index(self):
        """Rebuild the timeline vector index from database."""
        if not self._timeline_index:
            return

        cursor = self.conn.cursor()
        cursor.execute("SELECT id, embedding FROM timeline_events WHERE embedding IS NOT NULL")
        rows = cursor.fetchall()

        items = []
        for row in rows:
            embedding = self._deserialize_embedding(row["embedding"])
            if embedding:
                items.append((row["id"], embedding))

        if items:
            self._timeline_index.rebuild(items)
            self._timeline_index.save()

    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about vector indexes."""
        stats = {
            "use_vector_index": self._use_vector_index,
            "index_initialized": self._index_initialized
        }
        if self._memories_index:
            stats["memories"] = self._memories_index.get_stats()
        if self._patterns_index:
            stats["patterns"] = self._patterns_index.get_stats()
        if self._timeline_index:
            stats["timeline"] = self._timeline_index.get_stats()
        return stats

    async def initialize_schema(self):
        """Create necessary tables if they don't exist."""
        cursor = self.conn.cursor()

        # Main memories table with rich context
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Content
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT,

                -- Project Context
                project_path TEXT,
                project_name TEXT,
                project_type TEXT,
                tech_stack TEXT,

                -- Session Context
                session_id TEXT,
                chat_id TEXT,

                -- Agent/Skill Context
                agent_type TEXT,
                skill_used TEXT,
                tools_used TEXT,

                -- Outcome
                outcome TEXT,
                success INTEGER,
                user_feedback TEXT,

                -- Metadata
                tags TEXT,
                metadata TEXT DEFAULT '{}',
                importance INTEGER DEFAULT 5,

                -- Timestamps
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                last_accessed TEXT
            )
        """)

        # Projects table - store project-level knowledge
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                name TEXT,
                type TEXT,
                tech_stack TEXT,
                conventions TEXT,
                preferences TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # Patterns table - reusable solutions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                problem_type TEXT,
                solution TEXT NOT NULL,
                embedding TEXT,
                tech_context TEXT,
                success_count INTEGER DEFAULT 1,
                failure_count INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # Create indexes for memories/patterns
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_agent ON memories(agent_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_success ON memories(success)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_patterns_problem ON patterns(problem_type)")

        # Migration helper function
        def safe_add_column(table: str, column: str, column_def: str):
            """Safely add a column if it doesn't exist, with proper error handling."""
            try:
                cursor.execute(f"SELECT {column} FROM {table} LIMIT 1")
                logger.debug(f"Column {table}.{column} already exists")
            except sqlite3.OperationalError as e:
                if "no such column" in str(e).lower():
                    try:
                        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_def}")
                        logger.info(f"Migration: Added column {table}.{column}")
                    except sqlite3.OperationalError as alter_error:
                        if "duplicate column" not in str(alter_error).lower():
                            logger.error(f"Failed to add column {table}.{column}: {alter_error}")
                            raise MigrationError(
                                f"Failed to add column {table}.{column}",
                                original_error=alter_error
                            )
                else:
                    logger.error(f"Unexpected error checking column {table}.{column}: {e}")
                    raise MigrationError(
                        f"Unexpected error during migration check for {table}.{column}",
                        original_error=e
                    )
            except Exception as e:
                logger.error(f"Unexpected error in migration for {table}.{column}: {e}")
                raise MigrationError(
                    f"Migration failed for {table}.{column}",
                    original_error=e
                )

        # Migration: Add access_count column if it doesn't exist
        safe_add_column("memories", "access_count", "INTEGER DEFAULT 0")

        # Migration: Add decay_factor column if it doesn't exist
        safe_add_column("memories", "decay_factor", "REAL DEFAULT 1.0")

        # Migration: Add embedding_model column if it doesn't exist
        safe_add_column("memories", "embedding_model", "TEXT DEFAULT 'nomic-embed-text'")

        # Migration: Add confidence column if it doesn't exist
        # Confidence is a stored score (0.0 to 1.0) representing memory reliability
        # New memories start at 0.5, can be updated via API
        safe_add_column("memories", "confidence", "REAL DEFAULT 0.5")

        # Migration: Add outcome spectrum columns (v2.2.0)
        # These track the effectiveness of solutions stored as memories
        safe_add_column("memories", "outcome_status", "TEXT DEFAULT 'pending'")
        safe_add_column("memories", "fixed", "TEXT")  # JSON array of what was fixed
        safe_add_column("memories", "did_not_fix", "TEXT")  # JSON array of remaining issues
        safe_add_column("memories", "caused", "TEXT")  # JSON array of side effects
        safe_add_column("memories", "superseded_by", "INTEGER")  # FK to memories.id

        # Migration: Add self-correcting confidence columns (v2.2.1)
        # Track solution outcomes for automatic confidence adjustment
        safe_add_column("memories", "failure_count", "INTEGER DEFAULT 0")  # Consecutive failures
        safe_add_column("memories", "times_worked", "INTEGER DEFAULT 0")  # Total times solution worked
        safe_add_column("memories", "times_failed", "INTEGER DEFAULT 0")  # Total times solution failed

        # Migration: Add context tagging columns (v2.3.0)
        # Context-aware memory system - tracks where solutions worked/failed
        # This enables context-specific ranking: same solution may work in React but fail in Vue
        safe_add_column("memories", "worked_in", "TEXT")  # JSON array of contexts where solution worked
        safe_add_column("memories", "failed_in", "TEXT")  # JSON array of contexts where solution failed
        safe_add_column("memories", "context_confidence", "REAL")  # Context-specific confidence score

        # ============================================================
        # SESSION TIMELINE TABLES (Anti-Hallucination Layer)
        # ============================================================

        # Timeline events - chronological log of all session activity
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS timeline_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Session Context
                session_id TEXT NOT NULL,
                project_path TEXT,

                -- Event Identity
                event_type TEXT NOT NULL,
                sequence_num INTEGER NOT NULL,

                -- Content
                summary TEXT NOT NULL,
                details TEXT,
                embedding TEXT,

                -- Causal Chain
                parent_event_id INTEGER,
                root_event_id INTEGER,

                -- Entity References
                entities TEXT,

                -- Outcome
                status TEXT DEFAULT 'completed',
                outcome TEXT,
                confidence REAL,

                -- Flags
                is_anchor INTEGER DEFAULT 0,
                is_reversible INTEGER DEFAULT 1,
                needs_verification INTEGER DEFAULT 0,

                -- Timestamps
                created_at TEXT DEFAULT (datetime('now')),

                FOREIGN KEY (parent_event_id) REFERENCES timeline_events(id),
                FOREIGN KEY (root_event_id) REFERENCES timeline_events(id)
            )
        """)

        # Session state - current context for active session
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_state (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT UNIQUE NOT NULL,
                project_path TEXT,

                -- Current State
                current_goal TEXT,
                pending_questions TEXT,
                entity_registry TEXT,
                decisions_summary TEXT,

                -- Checkpoint tracking
                last_checkpoint_id INTEGER,
                events_since_checkpoint INTEGER DEFAULT 0,

                -- Timestamps
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                last_activity_at TEXT DEFAULT (datetime('now')),

                FOREIGN KEY (last_checkpoint_id) REFERENCES checkpoints(id)
            )
        """)

        # Checkpoints - session snapshots for resumption
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                event_id INTEGER,

                -- Checkpoint Content
                summary TEXT NOT NULL,
                key_facts TEXT,
                decisions TEXT,
                entities TEXT,

                -- State at Checkpoint
                current_goal TEXT,
                pending_items TEXT,

                -- For retrieval
                embedding TEXT,
                event_count INTEGER,

                -- Timestamps
                created_at TEXT DEFAULT (datetime('now')),

                FOREIGN KEY (event_id) REFERENCES timeline_events(id)
            )
        """)

        # Timeline indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeline_session ON timeline_events(session_id, sequence_num)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeline_type ON timeline_events(event_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeline_parent ON timeline_events(parent_event_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeline_root ON timeline_events(root_event_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeline_created ON timeline_events(created_at)")

        # Session state indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_project ON session_state(project_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_activity ON session_state(last_activity_at)")

        # Checkpoint indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_checkpoint_session ON checkpoints(session_id, created_at DESC)")

        # ============================================================
        # AGENT CONFIGURATION TABLES
        # ============================================================

        # Project agent configurations
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS project_agent_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_path TEXT NOT NULL,
                agent_id TEXT NOT NULL,
                enabled INTEGER DEFAULT 1,
                priority INTEGER DEFAULT 5,
                settings TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(project_path, agent_id)
            )
        """)

        # MCP server configurations per project
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS project_mcp_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_path TEXT NOT NULL,
                mcp_id TEXT NOT NULL,
                enabled INTEGER DEFAULT 1,
                settings TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(project_path, mcp_id)
            )
        """)

        # Hook configurations per project
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS project_hook_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_path TEXT NOT NULL,
                hook_id TEXT NOT NULL,
                enabled INTEGER DEFAULT 1,
                settings TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(project_path, hook_id)
            )
        """)

        # Project preferences
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS project_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_path TEXT UNIQUE NOT NULL,
                name TEXT,
                description TEXT,
                color TEXT DEFAULT '#58a6ff',
                icon TEXT DEFAULT 'folder',
                default_model TEXT DEFAULT 'sonnet',
                auto_memory INTEGER DEFAULT 1,
                auto_checkpoint INTEGER DEFAULT 1,
                settings TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # Agent config indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_agent_config_project ON project_agent_config(project_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mcp_config_project ON project_mcp_config(project_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_hook_config_project ON project_hook_config(project_path)")

        # ============================================================
        # INSIGHTS TABLE (Cross-Session Learning)
        # ============================================================

        # Aggregated insights from cross-session analysis
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,

                -- Insight Identity
                insight_type TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,

                -- Evidence
                evidence_ids TEXT,
                evidence_count INTEGER DEFAULT 1,
                source_sessions TEXT,

                -- Scoring
                confidence REAL DEFAULT 0.5,
                impact_score REAL DEFAULT 5.0,
                validation_count INTEGER DEFAULT 0,
                invalidation_count INTEGER DEFAULT 0,

                -- Categorization
                category TEXT,
                tags TEXT,
                project_path TEXT,
                tech_context TEXT,

                -- For similarity search
                embedding TEXT,

                -- Status
                status TEXT DEFAULT 'active',
                applied_to_claude_md INTEGER DEFAULT 0,

                -- Timestamps
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                last_validated_at TEXT
            )
        """)

        # Insight feedback for accuracy tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS insight_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                insight_id INTEGER NOT NULL,
                session_id TEXT,
                feedback_type TEXT NOT NULL,
                helpful INTEGER,
                comment TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (insight_id) REFERENCES insights(id)
            )
        """)

        # Insight indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_insights_type ON insights(insight_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_insights_status ON insights(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_insights_project ON insights(project_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_insights_confidence ON insights(confidence DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_insight_feedback ON insight_feedback(insight_id)")

        # ============================================================
        # MEMORY CLEANUP AND ARCHIVAL TABLES
        # ============================================================

        # Archived memories (soft-deleted for recovery)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_archive (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_id INTEGER NOT NULL,

                -- Original memory data
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT,
                project_path TEXT,
                session_id TEXT,
                importance INTEGER,
                access_count INTEGER,
                decay_factor REAL,
                metadata TEXT,

                -- Archive metadata
                archive_reason TEXT NOT NULL,
                archived_at TEXT DEFAULT (datetime('now')),
                archived_by TEXT,
                relevance_score_at_archive REAL,
                expires_at TEXT
            )
        """)

        # Cleanup configuration per project
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cleanup_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_path TEXT UNIQUE,

                -- Retention settings
                retention_days INTEGER DEFAULT 90,
                min_relevance_score REAL DEFAULT 0.1,
                keep_high_importance INTEGER DEFAULT 1,
                importance_threshold INTEGER DEFAULT 7,

                -- Deduplication settings
                dedup_enabled INTEGER DEFAULT 1,
                dedup_threshold REAL DEFAULT 0.95,

                -- Archive settings
                archive_before_delete INTEGER DEFAULT 1,
                archive_retention_days INTEGER DEFAULT 365,

                -- Schedule
                auto_cleanup_enabled INTEGER DEFAULT 0,
                last_cleanup_at TEXT,

                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # Cleanup audit log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cleanup_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cleanup_type TEXT NOT NULL,
                project_path TEXT,
                memories_archived INTEGER DEFAULT 0,
                memories_deleted INTEGER DEFAULT 0,
                memories_merged INTEGER DEFAULT 0,
                details TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # Archive indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_archive_original ON memory_archive(original_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_archive_project ON memory_archive(project_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_archive_reason ON memory_archive(archive_reason)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cleanup_project ON cleanup_config(project_path)")

        # ============================================================
        # ANCHOR CONFLICT RESOLUTION TABLES
        # ============================================================

        # Anchor conflicts for manual resolution
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anchor_conflicts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                project_path TEXT,

                -- The conflicting anchors
                anchor1_id INTEGER NOT NULL,
                anchor2_id INTEGER NOT NULL,
                anchor1_summary TEXT,
                anchor2_summary TEXT,

                -- Conflict details
                conflict_type TEXT NOT NULL,
                similarity_score REAL,
                auto_resolution_attempted INTEGER DEFAULT 0,

                -- Resolution
                status TEXT DEFAULT 'unresolved',
                resolution TEXT,
                resolved_anchor_id INTEGER,
                resolved_at TEXT,
                resolved_by TEXT,

                created_at TEXT DEFAULT (datetime('now')),

                FOREIGN KEY (anchor1_id) REFERENCES timeline_events(id),
                FOREIGN KEY (anchor2_id) REFERENCES timeline_events(id)
            )
        """)

        # Anchor history to track fact evolution
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anchor_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                anchor_id INTEGER NOT NULL,
                session_id TEXT,
                project_path TEXT,

                -- State tracking
                action TEXT NOT NULL,
                previous_summary TEXT,
                new_summary TEXT,
                superseded_by INTEGER,

                -- Context
                reason TEXT,
                confidence REAL,

                created_at TEXT DEFAULT (datetime('now')),

                FOREIGN KEY (anchor_id) REFERENCES timeline_events(id),
                FOREIGN KEY (superseded_by) REFERENCES timeline_events(id)
            )
        """)

        # Conflict indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conflicts_status ON anchor_conflicts(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_conflicts_session ON anchor_conflicts(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_anchor_history ON anchor_history(anchor_id)")

        # ============================================================
        # MARKDOWN SYNC TABLES (Moltbot-inspired transparency)
        # ============================================================

        # Markdown sync tracking - tracks which memories are synced to markdown files
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS markdown_syncs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_type TEXT NOT NULL,  -- 'memory_md', 'daily_log', 'flush'
                file_path TEXT NOT NULL,
                memory_id INTEGER,
                project_path TEXT,
                synced_at TEXT DEFAULT (datetime('now')),
                content_hash TEXT,

                FOREIGN KEY (memory_id) REFERENCES memories(id)
            )
        """)

        # Indexes for markdown_syncs
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_markdown_syncs_type ON markdown_syncs(file_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_markdown_syncs_project ON markdown_syncs(project_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_markdown_syncs_memory ON markdown_syncs(memory_id)")

        # ============================================================
        # KNOWLEDGE GRAPH RELATIONSHIPS TABLE
        # ============================================================

        # Memory relationships for knowledge graph traversal
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_relationships (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_id INTEGER NOT NULL,
                target_id INTEGER NOT NULL,
                relationship TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
                FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE,
                UNIQUE(source_id, target_id, relationship)
            )
        """)

        # Indexes for memory_relationships
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_source ON memory_relationships(source_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_target ON memory_relationships(target_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_rel_type ON memory_relationships(relationship)")

        # ============================================================
        # CURATOR AGENT TABLES
        # ============================================================

        # Curator configuration per project
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS curator_config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_path TEXT UNIQUE,
                auto_dedup_enabled INTEGER DEFAULT 1,
                auto_link_enabled INTEGER DEFAULT 1,
                dedup_threshold REAL DEFAULT 0.92,
                maintenance_interval_hours INTEGER DEFAULT 24,
                last_maintenance_at TEXT,
                curator_active INTEGER DEFAULT 1,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # Curator activity reports
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS curator_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_path TEXT,
                report_type TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                summary TEXT,
                findings TEXT,
                actions_taken TEXT,
                recommendations TEXT
            )
        """)

        # Curator indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_curator_config_project ON curator_config(project_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_curator_reports_project ON curator_reports(project_path)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_curator_reports_type ON curator_reports(report_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_curator_reports_created ON curator_reports(created_at DESC)")

        # Migration: Add last_flush_at column to session_state if it doesn't exist
        safe_add_column("session_state", "last_flush_at", "TEXT")

        self.conn.commit()

    def _serialize_embedding(self, embedding: List[float]) -> str:
        return json.dumps(embedding)

    def _deserialize_embedding(self, embedding_str: str) -> List[float]:
        return json.loads(embedding_str) if embedding_str else []

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        a = np.array(vec1)
        b = np.array(vec2)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def calculate_relevance_score(
        self,
        importance: int,
        created_at: str,
        last_accessed: Optional[str],
        access_count: int,
        decay_factor: float = 1.0,
        recency_half_life_days: float = 30.0
    ) -> float:
        """Calculate relevance score based on importance, recency, and access patterns.

        Formula: base_importance * recency_factor * access_factor * decay_factor

        Args:
            importance: Base importance (1-10)
            created_at: Creation timestamp
            last_accessed: Last access timestamp (None if never accessed)
            access_count: Number of times accessed
            decay_factor: Manual decay/boost multiplier
            recency_half_life_days: Days until score halves

        Returns:
            Relevance score (0.0 to ~10.0)
        """
        import math

        # Base importance (normalized to 0-1)
        base = importance / 10.0

        # Recency factor: exponential decay based on age
        now = datetime.now()
        try:
            # Use last_accessed if available, otherwise created_at
            reference_time = last_accessed or created_at
            if reference_time:
                # Parse timestamp (SQLite format: YYYY-MM-DD HH:MM:SS)
                ref_dt = datetime.fromisoformat(reference_time.replace('Z', '+00:00'))
                age_days = (now - ref_dt.replace(tzinfo=None)).days
                # Exponential decay: score halves every half_life_days
                recency_factor = math.pow(0.5, age_days / recency_half_life_days)
            else:
                recency_factor = 1.0
        except (ValueError, TypeError):
            recency_factor = 1.0

        # Access factor: boost frequently accessed memories (log scale)
        # +1 to avoid log(0), cap at reasonable value
        access_factor = 1.0 + 0.1 * math.log(1 + min(access_count, 100))

        # Combine factors
        score = base * recency_factor * access_factor * decay_factor

        return round(score, 4)

    async def update_access_stats(self, memory_id: int):
        """Update access statistics for a memory."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE memories
            SET last_accessed = datetime('now'),
                access_count = COALESCE(access_count, 0) + 1
            WHERE id = ?
            """,
            (memory_id,)
        )
        self.conn.commit()

    async def boost_memory(self, memory_id: int, factor: float = 1.5) -> bool:
        """Boost a memory's relevance by increasing its decay_factor.

        Args:
            memory_id: ID of the memory to boost
            factor: Multiplier to apply to current decay_factor

        Returns:
            True if successful
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE memories
            SET decay_factor = COALESCE(decay_factor, 1.0) * ?,
                updated_at = datetime('now')
            WHERE id = ?
            """,
            (factor, memory_id)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    async def decay_memory(self, memory_id: int, factor: float = 0.5) -> bool:
        """Reduce a memory's relevance by decreasing its decay_factor.

        Args:
            memory_id: ID of the memory to decay
            factor: Multiplier to apply to current decay_factor

        Returns:
            True if successful
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE memories
            SET decay_factor = COALESCE(decay_factor, 1.0) * ?,
                updated_at = datetime('now')
            WHERE id = ?
            """,
            (factor, memory_id)
        )
        self.conn.commit()
        return cursor.rowcount > 0

    async def get_memories_by_relevance(
        self,
        limit: int = 20,
        memory_type: Optional[str] = None,
        project_path: Optional[str] = None,
        min_relevance: float = 0.1
    ) -> List[Dict[str, Any]]:
        """Get memories sorted by relevance score.

        Args:
            limit: Maximum number of results
            memory_type: Filter by type
            project_path: Filter by project
            min_relevance: Minimum relevance score threshold

        Returns:
            List of memories with relevance scores
        """
        project_path = normalize_path(project_path)
        cursor = self.conn.cursor()

        query = """
            SELECT id, type, content, importance, created_at, last_accessed,
                   COALESCE(access_count, 0) as access_count,
                   COALESCE(decay_factor, 1.0) as decay_factor,
                   project_path, project_name, outcome, success
            FROM memories WHERE 1=1
        """
        params = []

        if memory_type:
            query += " AND type = ?"
            params.append(memory_type)
        if project_path:
            query += " AND project_path = ?"
            params.append(project_path)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            relevance = self.calculate_relevance_score(
                importance=row["importance"],
                created_at=row["created_at"],
                last_accessed=row["last_accessed"],
                access_count=row["access_count"],
                decay_factor=row["decay_factor"]
            )

            if relevance >= min_relevance:
                results.append({
                    "id": row["id"],
                    "type": row["type"],
                    "content": row["content"],
                    "relevance_score": relevance,
                    "importance": row["importance"],
                    "access_count": row["access_count"],
                    "decay_factor": row["decay_factor"],
                    "project_path": row["project_path"],
                    "outcome": row["outcome"],
                    "success": bool(row["success"]) if row["success"] is not None else None,
                    "created_at": row["created_at"],
                    "last_accessed": row["last_accessed"]
                })

        # Sort by relevance
        results.sort(key=lambda x: x["relevance_score"], reverse=True)
        return results[:limit]

    async def store_memory(
        self,
        memory_type: str,
        content: str,
        embedding: List[float],
        metadata: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
        # New context fields
        project_path: Optional[str] = None,
        project_name: Optional[str] = None,
        project_type: Optional[str] = None,
        tech_stack: Optional[List[str]] = None,
        chat_id: Optional[str] = None,
        agent_type: Optional[str] = None,
        skill_used: Optional[str] = None,
        tools_used: Optional[List[str]] = None,
        outcome: Optional[str] = None,
        success: Optional[bool] = None,
        tags: Optional[List[str]] = None,
        importance: int = 5,
        confidence: float = 0.5,  # Confidence score 0.0-1.0, default 0.5
        # Outcome spectrum fields
        outcome_status: Optional[str] = None,  # 'pending', 'success', 'partial', 'failed', 'superseded'
        fixed: Optional[List[str]] = None,
        did_not_fix: Optional[List[str]] = None,
        caused: Optional[List[str]] = None,
        superseded_by: Optional[int] = None,
        # Context tagging fields
        worked_in: Optional[List[Dict[str, Any]]] = None,
        failed_in: Optional[List[Dict[str, Any]]] = None,
        context_confidence: Optional[float] = None,
        auto_detect_context: bool = True
    ) -> int:
        """Store a memory with full context.

        Also adds the embedding to the FAISS index for fast search.

        Args:
            confidence: Reliability score from 0.0 (unreliable) to 1.0 (proven). Default 0.5.

        Outcome spectrum fields:
        - outcome_status: Status of the solution ('pending', 'success', 'partial', 'failed', 'superseded')
        - fixed: List of what this solution fixed
        - did_not_fix: List of what remains unfixed
        - caused: List of side effects this solution caused
        - superseded_by: ID of memory that replaced this one

        Context tagging fields:
        - worked_in: List of contexts where this solution worked
        - failed_in: List of contexts where this solution failed
        - context_confidence: Context-specific confidence score
        - auto_detect_context: If True and project_path provided, auto-detect context
        """
        # Normalize project path to prevent duplicates
        project_path = normalize_path(project_path)

        # Auto-detect context from project_path if enabled
        initial_context = None
        if auto_detect_context and project_path:
            try:
                from skills.context import detect_project_context
                initial_context = detect_project_context(project_path)
                # Set worked_in to initial context if success is True
                if success is True and initial_context and not worked_in:
                    worked_in = [initial_context]
                # Set failed_in to initial context if success is False
                elif success is False and initial_context and not failed_in:
                    failed_in = [initial_context]
            except Exception:
                pass  # Ignore context detection errors

        # Default outcome_status to 'pending' for new memories
        if outcome_status is None:
            outcome_status = 'pending'

        # Clamp confidence to valid range
        confidence = max(0.0, min(1.0, confidence))

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO memories (
                type, content, embedding, metadata,
                project_path, project_name, project_type, tech_stack,
                session_id, chat_id,
                agent_type, skill_used, tools_used,
                outcome, success,
                tags, importance, confidence,
                outcome_status, fixed, did_not_fix, caused, superseded_by,
                worked_in, failed_in, context_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory_type,
                content,
                self._serialize_embedding(embedding),
                json.dumps(metadata or {}),
                project_path,
                project_name,
                project_type,
                json.dumps(tech_stack) if tech_stack else None,
                session_id,
                chat_id,
                agent_type,
                skill_used,
                json.dumps(tools_used) if tools_used else None,
                outcome,
                1 if success else (0 if success is False else None),
                json.dumps(tags) if tags else None,
                importance,
                confidence,
                outcome_status,
                json.dumps(fixed) if fixed else None,
                json.dumps(did_not_fix) if did_not_fix else None,
                json.dumps(caused) if caused else None,
                superseded_by,
                json.dumps(worked_in) if worked_in else None,
                json.dumps(failed_in) if failed_in else None,
                context_confidence
            )
        )
        self.conn.commit()
        memory_id = cursor.lastrowid

        # Add to FAISS index if available
        if self._memories_index and embedding:
            self._memories_index.add(memory_id, embedding)

        return memory_id

    async def search_similar(
        self,
        embedding: List[float],
        limit: int = 10,
        memory_type: Optional[str] = None,
        session_id: Optional[str] = None,
        project_path: Optional[str] = None,
        agent_type: Optional[str] = None,
        success_only: bool = False,
        threshold: float = 0.5,
        # Outcome spectrum filters
        include_failed: bool = False,
        include_superseded: bool = False,
        include_unreliable: bool = False,
        outcome_status: Optional[str] = None,
        # Context-aware search
        current_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar memories with optional filters.

        Uses FAISS index for fast similarity search when available,
        falls back to numpy-based linear search otherwise.

        Outcome-aware search behavior:
        - 'success' memories rank highest (1.5x boost)
        - 'partial' memories shown with warning (1.0x - no penalty)
        - 'failed' memories excluded by default (use include_failed=True to show)
        - 'superseded' memories excluded and replaced with their superseding memory
        - 'pending' memories shown normally (1.0x)
        - Memories with failure_count >= 3 are excluded by default (use include_unreliable=True)

        Context-aware search:
        - If current_context provided, memories that worked in similar contexts get +0.2 boost
        - Memories that failed in similar contexts get -0.2 penalty
        """
        # Normalize project path for consistent matching
        project_path = normalize_path(project_path)

        # Ensure indexes are initialized
        await self._init_vector_indexes()

        cursor = self.conn.cursor()
        has_filters = memory_type or session_id or project_path or agent_type or success_only

        # Try FAISS index first (if no filters or willing to post-filter)
        if self._memories_index and self._memories_index.size() > 0:
            # Get more candidates than needed to allow for filtering
            candidate_limit = limit * 5 if has_filters else limit * 2

            # FAISS search
            candidates = self._memories_index.search(
                query_embedding=embedding,
                k=candidate_limit,
                threshold=threshold
            )

            if candidates:
                # Get full records for candidates
                candidate_ids = [c[0] for c in candidates]
                similarity_map = {c[0]: c[1] for c in candidates}

                # Build query with filters
                placeholders = ",".join("?" * len(candidate_ids))
                query = f"""
                    SELECT id, type, content, metadata,
                           project_path, project_name, project_type, tech_stack,
                           session_id, chat_id, agent_type, skill_used, tools_used,
                           outcome, success, tags, importance, confidence, created_at,
                           outcome_status, fixed, did_not_fix, caused, superseded_by,
                           worked_in, failed_in, context_confidence
                    FROM memories WHERE id IN ({placeholders})
                """
                params = list(candidate_ids)

                if memory_type:
                    query += " AND type = ?"
                    params.append(memory_type)
                if session_id:
                    query += " AND session_id = ?"
                    params.append(session_id)
                if project_path:
                    query += " AND project_path = ?"
                    params.append(project_path)
                if agent_type:
                    query += " AND agent_type = ?"
                    params.append(agent_type)
                if success_only:
                    query += " AND success = 1"

                # Outcome spectrum filters
                if outcome_status:
                    query += " AND outcome_status = ?"
                    params.append(outcome_status)
                else:
                    # Default behavior: exclude failed and superseded
                    if not include_failed:
                        query += " AND (outcome_status IS NULL OR outcome_status != 'failed')"
                    if not include_superseded:
                        query += " AND (outcome_status IS NULL OR outcome_status != 'superseded')"

                # Exclude unreliable memories (failure_count >= 3) by default
                if not include_unreliable:
                    query += " AND (failure_count IS NULL OR failure_count < 3)"

                cursor.execute(query, params)
                rows = cursor.fetchall()

                results = []
                superseding_memories = {}  # Cache for superseding memories

                # Import context scoring if current_context provided
                context_scorer = None
                if current_context:
                    try:
                        from skills.context import calculate_context_similarity
                        context_scorer = calculate_context_similarity
                    except ImportError:
                        pass

                for row in rows:
                    similarity = similarity_map.get(row["id"], 0)

                    # Calculate outcome-based ranking boost
                    row_outcome_status = row["outcome_status"] if "outcome_status" in row.keys() else None
                    outcome_boost = 1.0
                    outcome_warning = None
                    if row_outcome_status == 'success':
                        outcome_boost = 1.5  # Boost successful solutions
                    elif row_outcome_status == 'partial':
                        outcome_warning = "This solution only partially worked"
                    elif row_outcome_status == 'failed':
                        outcome_boost = 0.5  # Penalize failed solutions
                        outcome_warning = "This solution failed previously"

                    # Calculate context-based adjustment
                    context_adjustment = 0.0
                    context_recommendation = None
                    if context_scorer and current_context:
                        worked_in = json.loads(row["worked_in"]) if ("worked_in" in row.keys() and row["worked_in"]) else []
                        failed_in = json.loads(row["failed_in"]) if ("failed_in" in row.keys() and row["failed_in"]) else []

                        # Calculate similarity to worked_in contexts (boost)
                        max_success_sim = 0.0
                        for ctx in worked_in:
                            sim = context_scorer(current_context, ctx)
                            max_success_sim = max(max_success_sim, sim)

                        # Calculate similarity to failed_in contexts (penalty)
                        max_failure_sim = 0.0
                        for ctx in failed_in:
                            sim = context_scorer(current_context, ctx)
                            max_failure_sim = max(max_failure_sim, sim)

                        # Context adjustment: +0.2 for worked_in match, -0.2 for failed_in match
                        context_adjustment = (max_success_sim * 0.2) - (max_failure_sim * 0.2)

                        if context_adjustment > 0.1:
                            context_recommendation = "recommended_for_context"
                        elif context_adjustment < -0.1:
                            context_recommendation = "caution_different_context"

                    results.append({
                        "id": row["id"],
                        "type": row["type"],
                        "content": row["content"],
                        "similarity": similarity,
                        "search_method": "faiss",
                        "project": {
                            "path": row["project_path"],
                            "name": row["project_name"],
                            "type": row["project_type"],
                            "tech_stack": json.loads(row["tech_stack"]) if row["tech_stack"] else None
                        },
                        "session_id": row["session_id"],
                        "agent": {
                            "type": row["agent_type"],
                            "skill": row["skill_used"],
                            "tools": json.loads(row["tools_used"]) if row["tools_used"] else None
                        },
                        "outcome": row["outcome"],
                        "success": bool(row["success"]) if row["success"] is not None else None,
                        "outcome_status": row_outcome_status,
                        "outcome_boost": outcome_boost,
                        "outcome_warning": outcome_warning,
                        "context_adjustment": context_adjustment,
                        "context_recommendation": context_recommendation,
                        "fixed": json.loads(row["fixed"]) if ("fixed" in row.keys() and row["fixed"]) else None,
                        "did_not_fix": json.loads(row["did_not_fix"]) if ("did_not_fix" in row.keys() and row["did_not_fix"]) else None,
                        "caused": json.loads(row["caused"]) if ("caused" in row.keys() and row["caused"]) else None,
                        "superseded_by": row["superseded_by"] if "superseded_by" in row.keys() else None,
                        "worked_in": json.loads(row["worked_in"]) if ("worked_in" in row.keys() and row["worked_in"]) else None,
                        "failed_in": json.loads(row["failed_in"]) if ("failed_in" in row.keys() and row["failed_in"]) else None,
                        "context_confidence": row["context_confidence"] if "context_confidence" in row.keys() else None,
                        "tags": json.loads(row["tags"]) if row["tags"] else None,
                        "importance": row["importance"],
                        "confidence": row["confidence"] if row["confidence"] is not None else 0.5,
                        "created_at": row["created_at"],
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
                    })

                # Sort by combined score: (similarity * 0.7) + (confidence * 0.3) + context_adjustment
                # This ranking prioritizes semantic relevance while boosting high-confidence memories
                # and adjusting for context compatibility
                results.sort(
                    key=lambda x: (x["similarity"] * 0.7) + (x["confidence"] * 0.3) + x.get("context_adjustment", 0.0),
                    reverse=True
                )

                # Update last_accessed for returned results
                if results:
                    ids = [r["id"] for r in results[:limit]]
                    cursor.execute(
                        f"UPDATE memories SET last_accessed = datetime('now') WHERE id IN ({','.join('?' * len(ids))})",
                        ids
                    )
                    self.conn.commit()

                return results[:limit]

        # Fallback to numpy-based search (original implementation)
        query = """
            SELECT id, type, content, embedding, metadata,
                   project_path, project_name, project_type, tech_stack,
                   session_id, chat_id, agent_type, skill_used, tools_used,
                   outcome, success, tags, importance, confidence, created_at,
                   outcome_status, fixed, did_not_fix, caused, superseded_by,
                   worked_in, failed_in, context_confidence
            FROM memories WHERE 1=1
        """
        params = []

        if memory_type:
            query += " AND type = ?"
            params.append(memory_type)
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if project_path:
            query += " AND project_path = ?"
            params.append(project_path)
        if agent_type:
            query += " AND agent_type = ?"
            params.append(agent_type)
        if success_only:
            query += " AND success = 1"

        # Outcome spectrum filters for numpy fallback
        if outcome_status:
            query += " AND outcome_status = ?"
            params.append(outcome_status)
        else:
            if not include_failed:
                query += " AND (outcome_status IS NULL OR outcome_status != 'failed')"
            if not include_superseded:
                query += " AND (outcome_status IS NULL OR outcome_status != 'superseded')"

        # Exclude unreliable memories (failure_count >= 3) by default
        if not include_unreliable:
            query += " AND (failure_count IS NULL OR failure_count < 3)"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []

        # Import context scoring if current_context provided
        context_scorer = None
        if current_context:
            try:
                from skills.context import calculate_context_similarity
                context_scorer = calculate_context_similarity
            except ImportError:
                pass

        for row in rows:
            stored_embedding = self._deserialize_embedding(row["embedding"])
            if stored_embedding:
                similarity = self._cosine_similarity(embedding, stored_embedding)
                if similarity >= threshold:
                    # Calculate outcome-based ranking boost
                    row_outcome_status = row["outcome_status"] if "outcome_status" in row.keys() else None
                    outcome_boost = 1.0
                    outcome_warning = None
                    if row_outcome_status == 'success':
                        outcome_boost = 1.5
                    elif row_outcome_status == 'partial':
                        outcome_warning = "This solution only partially worked"
                    elif row_outcome_status == 'failed':
                        outcome_boost = 0.5
                        outcome_warning = "This solution failed previously"

                    # Calculate context-based adjustment
                    context_adjustment = 0.0
                    context_recommendation = None
                    if context_scorer and current_context:
                        worked_in = json.loads(row["worked_in"]) if ("worked_in" in row.keys() and row["worked_in"]) else []
                        failed_in = json.loads(row["failed_in"]) if ("failed_in" in row.keys() and row["failed_in"]) else []

                        max_success_sim = 0.0
                        for ctx in worked_in:
                            sim = context_scorer(current_context, ctx)
                            max_success_sim = max(max_success_sim, sim)

                        max_failure_sim = 0.0
                        for ctx in failed_in:
                            sim = context_scorer(current_context, ctx)
                            max_failure_sim = max(max_failure_sim, sim)

                        context_adjustment = (max_success_sim * 0.2) - (max_failure_sim * 0.2)

                        if context_adjustment > 0.1:
                            context_recommendation = "recommended_for_context"
                        elif context_adjustment < -0.1:
                            context_recommendation = "caution_different_context"

                    results.append({
                        "id": row["id"],
                        "type": row["type"],
                        "content": row["content"],
                        "similarity": similarity,
                        "search_method": "numpy",
                        "project": {
                            "path": row["project_path"],
                            "name": row["project_name"],
                            "type": row["project_type"],
                            "tech_stack": json.loads(row["tech_stack"]) if row["tech_stack"] else None
                        },
                        "session_id": row["session_id"],
                        "agent": {
                            "type": row["agent_type"],
                            "skill": row["skill_used"],
                            "tools": json.loads(row["tools_used"]) if row["tools_used"] else None
                        },
                        "outcome": row["outcome"],
                        "success": bool(row["success"]) if row["success"] is not None else None,
                        "outcome_status": row_outcome_status,
                        "outcome_boost": outcome_boost,
                        "outcome_warning": outcome_warning,
                        "context_adjustment": context_adjustment,
                        "context_recommendation": context_recommendation,
                        "fixed": json.loads(row["fixed"]) if ("fixed" in row.keys() and row["fixed"]) else None,
                        "did_not_fix": json.loads(row["did_not_fix"]) if ("did_not_fix" in row.keys() and row["did_not_fix"]) else None,
                        "caused": json.loads(row["caused"]) if ("caused" in row.keys() and row["caused"]) else None,
                        "superseded_by": row["superseded_by"] if "superseded_by" in row.keys() else None,
                        "worked_in": json.loads(row["worked_in"]) if ("worked_in" in row.keys() and row["worked_in"]) else None,
                        "failed_in": json.loads(row["failed_in"]) if ("failed_in" in row.keys() and row["failed_in"]) else None,
                        "context_confidence": row["context_confidence"] if "context_confidence" in row.keys() else None,
                        "tags": json.loads(row["tags"]) if row["tags"] else None,
                        "importance": row["importance"],
                        "confidence": row["confidence"] if row["confidence"] is not None else 0.5,
                        "created_at": row["created_at"],
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
                    })

        # Sort by combined score including outcome boost and context adjustment
        results.sort(
            key=lambda x: ((x["similarity"] * 0.7) + (x["confidence"] * 0.3) + x.get("context_adjustment", 0.0)) * x.get("outcome_boost", 1.0),
            reverse=True
        )

        # Update last_accessed for returned results
        if results:
            ids = [r["id"] for r in results[:limit]]
            cursor.execute(
                f"UPDATE memories SET last_accessed = datetime('now') WHERE id IN ({','.join('?' * len(ids))})",
                ids
            )
            self.conn.commit()

        return results[:limit]

    async def update_memory_confidence(
        self,
        memory_id: int,
        confidence: float
    ) -> Dict[str, Any]:
        """Update the confidence score for a memory.

        Args:
            memory_id: ID of the memory to update
            confidence: New confidence score (0.0 to 1.0)

        Returns:
            Dict with success status and updated confidence
        """
        # Clamp confidence to valid range
        confidence = max(0.0, min(1.0, confidence))

        cursor = self.conn.cursor()

        # Check if memory exists
        cursor.execute("SELECT id, confidence FROM memories WHERE id = ?", [memory_id])
        row = cursor.fetchone()

        if not row:
            return {
                "success": False,
                "error": f"Memory with ID {memory_id} not found"
            }

        old_confidence = row["confidence"] if row["confidence"] is not None else 0.5

        # Update confidence
        cursor.execute(
            "UPDATE memories SET confidence = ?, updated_at = datetime('now') WHERE id = ?",
            [confidence, memory_id]
        )
        self.conn.commit()

        return {
            "success": True,
            "memory_id": memory_id,
            "old_confidence": old_confidence,
            "new_confidence": confidence,
            "message": f"Confidence updated from {old_confidence:.3f} to {confidence:.3f}"
        }

    async def keyword_search(
        self,
        query: str,
        limit: int = 10,
        memory_type: Optional[str] = None,
        session_id: Optional[str] = None,
        project_path: Optional[str] = None,
        agent_type: Optional[str] = None,
        success_only: bool = False,
        include_failed: bool = False,
        include_superseded: bool = False,
        include_unreliable: bool = False,
        outcome_status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fallback keyword search when embeddings are unavailable.

        Uses SQLite FTS-like matching with LIKE queries on content.
        Results are ranked by keyword match count and importance.
        Supports outcome spectrum filtering.
        Excludes unreliable memories (failure_count >= 3) by default.
        """
        # Normalize project path for consistent matching
        project_path = normalize_path(project_path)

        cursor = self.conn.cursor()

        # Extract keywords from query (simple tokenization)
        keywords = [k.strip().lower() for k in query.split() if len(k.strip()) >= 3]
        if not keywords:
            keywords = [query.lower()]

        # Build query with keyword matching
        sql = """
            SELECT id, type, content, metadata,
                   project_path, project_name, project_type, tech_stack,
                   session_id, chat_id, agent_type, skill_used, tools_used,
                   outcome, success, tags, importance, confidence, created_at,
                   outcome_status, fixed, did_not_fix, caused, superseded_by
            FROM memories WHERE 1=1
        """
        params = []

        if memory_type:
            sql += " AND type = ?"
            params.append(memory_type)
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        if project_path:
            sql += " AND project_path = ?"
            params.append(project_path)
        if agent_type:
            sql += " AND agent_type = ?"
            params.append(agent_type)
        if success_only:
            sql += " AND success = 1"

        # Outcome spectrum filters
        if outcome_status:
            sql += " AND outcome_status = ?"
            params.append(outcome_status)
        else:
            if not include_failed:
                sql += " AND (outcome_status IS NULL OR outcome_status != 'failed')"
            if not include_superseded:
                sql += " AND (outcome_status IS NULL OR outcome_status != 'superseded')"

        # Exclude unreliable memories (failure_count >= 3) by default
        if not include_unreliable:
            sql += " AND (failure_count IS NULL OR failure_count < 3)"

        # Add keyword conditions (OR matching)
        keyword_conditions = []
        for kw in keywords:
            keyword_conditions.append("LOWER(content) LIKE ?")
            params.append(f"%{kw}%")

        if keyword_conditions:
            sql += f" AND ({' OR '.join(keyword_conditions)})"

        sql += " ORDER BY importance DESC, created_at DESC"
        sql += f" LIMIT {limit * 3}"  # Get more for ranking

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            content_lower = row["content"].lower()
            # Calculate keyword match score
            match_count = sum(1 for kw in keywords if kw in content_lower)
            keyword_score = match_count / len(keywords) if keywords else 0

            # Calculate outcome-based ranking boost
            row_outcome_status = row["outcome_status"] if "outcome_status" in row.keys() else None
            outcome_boost = 1.0
            outcome_warning = None
            if row_outcome_status == 'success':
                outcome_boost = 1.5
            elif row_outcome_status == 'partial':
                outcome_warning = "This solution only partially worked"
            elif row_outcome_status == 'failed':
                outcome_boost = 0.5
                outcome_warning = "This solution failed previously"

            results.append({
                "id": row["id"],
                "type": row["type"],
                "content": row["content"],
                "similarity": keyword_score,  # Use keyword score as pseudo-similarity
                "match_type": "keyword",
                "keywords_matched": match_count,
                "project": {
                    "path": row["project_path"],
                    "name": row["project_name"],
                    "type": row["project_type"],
                    "tech_stack": json.loads(row["tech_stack"]) if row["tech_stack"] else None
                },
                "session_id": row["session_id"],
                "agent": {
                    "type": row["agent_type"],
                    "skill": row["skill_used"],
                    "tools": json.loads(row["tools_used"]) if row["tools_used"] else None
                },
                "outcome": row["outcome"],
                "success": bool(row["success"]) if row["success"] is not None else None,
                "outcome_status": row_outcome_status,
                "outcome_boost": outcome_boost,
                "outcome_warning": outcome_warning,
                "fixed": json.loads(row["fixed"]) if ("fixed" in row.keys() and row["fixed"]) else None,
                "did_not_fix": json.loads(row["did_not_fix"]) if ("did_not_fix" in row.keys() and row["did_not_fix"]) else None,
                "caused": json.loads(row["caused"]) if ("caused" in row.keys() and row["caused"]) else None,
                "superseded_by": row["superseded_by"] if "superseded_by" in row.keys() else None,
                "tags": json.loads(row["tags"]) if row["tags"] else None,
                "importance": row["importance"],
                "confidence": row["confidence"] if ("confidence" in row.keys() and row["confidence"] is not None) else 0.5,
                "created_at": row["created_at"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
            })

        # Sort by: (keyword_score * 0.7) + (confidence * 0.3) * outcome_boost
        results.sort(
            key=lambda x: ((x["similarity"] * 0.7) + (x["confidence"] * 0.3)) * x.get("outcome_boost", 1.0),
            reverse=True
        )

        # Update last_accessed for returned results
        if results:
            ids = [r["id"] for r in results[:limit]]
            cursor.execute(
                f"UPDATE memories SET last_accessed = datetime('now') WHERE id IN ({','.join('?' * len(ids))})",
                ids
            )
            self.conn.commit()

        return results[:limit]

    async def store_project(
        self,
        path: str,
        name: Optional[str] = None,
        project_type: Optional[str] = None,
        tech_stack: Optional[List[str]] = None,
        conventions: Optional[Dict[str, Any]] = None,
        preferences: Optional[Dict[str, Any]] = None
    ) -> int:
        """Store or update project information."""
        # Normalize path to prevent duplicates
        path = normalize_path(path)

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO projects (path, name, type, tech_stack, conventions, preferences)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                name = excluded.name,
                type = excluded.type,
                tech_stack = excluded.tech_stack,
                conventions = excluded.conventions,
                preferences = excluded.preferences,
                updated_at = datetime('now')
            """,
            (
                path,
                name,
                project_type,
                json.dumps(tech_stack) if tech_stack else None,
                json.dumps(conventions) if conventions else None,
                json.dumps(preferences) if preferences else None
            )
        )
        self.conn.commit()
        return cursor.lastrowid

    async def get_project(self, path: str) -> Optional[Dict[str, Any]]:
        """Get project information."""
        # Normalize path for consistent matching
        path = normalize_path(path)

        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM projects WHERE path = ?", (path,))
        row = cursor.fetchone()
        if row:
            return {
                "id": row["id"],
                "path": row["path"],
                "name": row["name"],
                "type": row["type"],
                "tech_stack": json.loads(row["tech_stack"]) if row["tech_stack"] else None,
                "conventions": json.loads(row["conventions"]) if row["conventions"] else None,
                "preferences": json.loads(row["preferences"]) if row["preferences"] else None
            }
        return None

    async def store_pattern(
        self,
        name: str,
        solution: str,
        embedding: List[float],
        problem_type: Optional[str] = None,
        tech_context: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Store a reusable pattern/solution."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO patterns (name, problem_type, solution, embedding, tech_context, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                name,
                problem_type,
                solution,
                self._serialize_embedding(embedding),
                json.dumps(tech_context) if tech_context else None,
                json.dumps(metadata or {})
            )
        )
        self.conn.commit()
        return cursor.lastrowid

    async def search_patterns(
        self,
        embedding: List[float],
        limit: int = 5,
        problem_type: Optional[str] = None,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search for similar patterns."""
        cursor = self.conn.cursor()

        query = "SELECT * FROM patterns WHERE 1=1"
        params = []
        if problem_type:
            query += " AND problem_type = ?"
            params.append(problem_type)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            stored_embedding = self._deserialize_embedding(row["embedding"])
            if stored_embedding:
                similarity = self._cosine_similarity(embedding, stored_embedding)
                if similarity >= threshold:
                    # Weight by success rate
                    total = row["success_count"] + row["failure_count"]
                    success_rate = row["success_count"] / total if total > 0 else 0.5

                    results.append({
                        "id": row["id"],
                        "name": row["name"],
                        "problem_type": row["problem_type"],
                        "solution": row["solution"],
                        "tech_context": json.loads(row["tech_context"]) if row["tech_context"] else None,
                        "similarity": similarity,
                        "success_rate": success_rate,
                        "score": similarity * success_rate
                    })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    async def keyword_search_patterns(
        self,
        query: str,
        limit: int = 5,
        problem_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fallback keyword search for patterns when embeddings unavailable."""
        cursor = self.conn.cursor()

        # Extract keywords from query
        keywords = [k.strip().lower() for k in query.split() if len(k.strip()) >= 3]
        if not keywords:
            keywords = [query.lower()]

        sql = "SELECT * FROM patterns WHERE 1=1"
        params = []

        if problem_type:
            sql += " AND problem_type = ?"
            params.append(problem_type)

        # Add keyword conditions
        keyword_conditions = []
        for kw in keywords:
            keyword_conditions.append("(LOWER(name) LIKE ? OR LOWER(solution) LIKE ?)")
            params.append(f"%{kw}%")
            params.append(f"%{kw}%")

        if keyword_conditions:
            sql += f" AND ({' OR '.join(keyword_conditions)})"

        sql += " ORDER BY success_count DESC, created_at DESC"
        sql += f" LIMIT {limit * 2}"

        cursor.execute(sql, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            # Calculate keyword match score
            combined_text = f"{row['name']} {row['solution']}".lower()
            match_count = sum(1 for kw in keywords if kw in combined_text)
            keyword_score = match_count / len(keywords) if keywords else 0

            total = row["success_count"] + row["failure_count"]
            success_rate = row["success_count"] / total if total > 0 else 0.5

            results.append({
                "id": row["id"],
                "name": row["name"],
                "problem_type": row["problem_type"],
                "solution": row["solution"],
                "tech_context": json.loads(row["tech_context"]) if row["tech_context"] else None,
                "similarity": keyword_score,
                "match_type": "keyword",
                "keywords_matched": match_count,
                "success_rate": success_rate,
                "score": keyword_score * success_rate
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    async def update_pattern_outcome(self, pattern_id: int, success: bool):
        """Update pattern success/failure count."""
        cursor = self.conn.cursor()
        if success:
            cursor.execute("UPDATE patterns SET success_count = success_count + 1 WHERE id = ?", (pattern_id,))
        else:
            cursor.execute("UPDATE patterns SET failure_count = failure_count + 1 WHERE id = ?", (pattern_id,))
        self.conn.commit()

    async def update_memory_outcome(
        self,
        memory_id: int,
        outcome_status: Optional[str] = None,
        fixed: Optional[List[str]] = None,
        did_not_fix: Optional[List[str]] = None,
        caused: Optional[List[str]] = None,
        superseded_by: Optional[int] = None
    ) -> Dict[str, Any]:
        """Update the outcome status and details for a memory.

        Args:
            memory_id: The ID of the memory to update
            outcome_status: New status ('pending', 'success', 'partial', 'failed', 'superseded')
            fixed: List of what this solution fixed (appends to existing)
            did_not_fix: List of what remains unfixed (appends to existing)
            caused: List of side effects (appends to existing)
            superseded_by: ID of the memory that replaces this one

        Returns:
            Dict with update status and updated memory info
        """
        valid_statuses = {'pending', 'success', 'partial', 'failed', 'superseded'}
        if outcome_status and outcome_status not in valid_statuses:
            raise ValueError(f"Invalid outcome_status: {outcome_status}. Must be one of {valid_statuses}")

        cursor = self.conn.cursor()

        # Get current memory state
        cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        if not row:
            return {"success": False, "error": f"Memory {memory_id} not found"}

        # Build update query
        updates = []
        params = []

        if outcome_status:
            updates.append("outcome_status = ?")
            params.append(outcome_status)
            # Also update the legacy success field for compatibility
            if outcome_status == 'success':
                updates.append("success = 1")
            elif outcome_status == 'failed':
                updates.append("success = 0")

        # For list fields, merge with existing
        if fixed:
            existing = json.loads(row["fixed"]) if row["fixed"] else []
            merged = list(set(existing + fixed))
            updates.append("fixed = ?")
            params.append(json.dumps(merged))

        if did_not_fix:
            existing = json.loads(row["did_not_fix"]) if row["did_not_fix"] else []
            merged = list(set(existing + did_not_fix))
            updates.append("did_not_fix = ?")
            params.append(json.dumps(merged))

        if caused:
            existing = json.loads(row["caused"]) if row["caused"] else []
            merged = list(set(existing + caused))
            updates.append("caused = ?")
            params.append(json.dumps(merged))

        if superseded_by is not None:
            updates.append("superseded_by = ?")
            params.append(superseded_by)
            # Auto-set status to superseded if not explicitly set
            if not outcome_status:
                updates.append("outcome_status = 'superseded'")

        updates.append("updated_at = datetime('now')")

        if updates:
            query = f"UPDATE memories SET {', '.join(updates)} WHERE id = ?"
            params.append(memory_id)
            cursor.execute(query, params)
            self.conn.commit()

        return {
            "success": True,
            "memory_id": memory_id,
            "outcome_status": outcome_status or row["outcome_status"],
            "message": f"Memory {memory_id} outcome updated"
        }

    async def supersede_memory(
        self,
        old_memory_id: int,
        new_memory_id: int,
        reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """Mark a memory as superseded by another memory.

        This is a convenience method that:
        1. Sets the old memory's status to 'superseded'
        2. Sets its superseded_by field to point to the new memory
        3. Optionally stores the reason in metadata

        Args:
            old_memory_id: The memory being replaced
            new_memory_id: The memory that replaces it
            reason: Optional reason for supersession

        Returns:
            Dict with operation status
        """
        cursor = self.conn.cursor()

        # Verify both memories exist
        cursor.execute("SELECT id FROM memories WHERE id = ?", (old_memory_id,))
        if not cursor.fetchone():
            return {"success": False, "error": f"Memory {old_memory_id} not found"}

        cursor.execute("SELECT id FROM memories WHERE id = ?", (new_memory_id,))
        if not cursor.fetchone():
            return {"success": False, "error": f"Memory {new_memory_id} not found"}

        # Update the old memory
        update_result = await self.update_memory_outcome(
            memory_id=old_memory_id,
            outcome_status='superseded',
            superseded_by=new_memory_id
        )

        if not update_result.get("success"):
            return update_result

        # Store reason in metadata if provided
        if reason:
            cursor.execute("SELECT metadata FROM memories WHERE id = ?", (old_memory_id,))
            row = cursor.fetchone()
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            metadata["supersession_reason"] = reason
            metadata["superseded_at"] = datetime.now().isoformat()
            cursor.execute(
                "UPDATE memories SET metadata = ? WHERE id = ?",
                (json.dumps(metadata), old_memory_id)
            )
            self.conn.commit()

        return {
            "success": True,
            "old_memory_id": old_memory_id,
            "new_memory_id": new_memory_id,
            "reason": reason,
            "message": f"Memory {old_memory_id} superseded by {new_memory_id}"
        }

    async def get_superseding_memory(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Get the memory that supersedes the given memory, if any.

        Follows the supersession chain to find the latest active memory.

        Returns:
            The latest superseding memory, or None if not superseded
        """
        cursor = self.conn.cursor()
        visited = set()
        current_id = memory_id

        while True:
            if current_id in visited:
                # Circular reference detected
                logger.warning(f"Circular supersession detected at memory {current_id}")
                break
            visited.add(current_id)

            cursor.execute(
                "SELECT id, superseded_by, outcome_status FROM memories WHERE id = ?",
                (current_id,)
            )
            row = cursor.fetchone()
            if not row:
                break

            if row["superseded_by"] and row["outcome_status"] == 'superseded':
                current_id = row["superseded_by"]
            else:
                # Found the latest non-superseded memory
                if current_id != memory_id:
                    return await self.get_memory(current_id)
                return None

        return None

    async def get_memory(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM memories WHERE id = ?", (memory_id,))
        row = cursor.fetchone()
        if row:
            return {
                "id": row["id"],
                "type": row["type"],
                "content": row["content"],
                "project": {
                    "path": row["project_path"],
                    "name": row["project_name"],
                    "type": row["project_type"]
                },
                "session_id": row["session_id"],
                "agent_type": row["agent_type"],
                "skill_used": row["skill_used"],
                "outcome": row["outcome"],
                "success": bool(row["success"]) if row["success"] is not None else None,
                "importance": row["importance"],
                "created_at": row["created_at"],
                "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
            }
        return None

    async def get_memories_by_type(
        self,
        memory_type: str,
        limit: int = 50,
        session_id: Optional[str] = None,
        project_path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve memories by type."""
        # Normalize project path for consistent matching
        project_path = normalize_path(project_path)

        cursor = self.conn.cursor()

        query = "SELECT * FROM memories WHERE type = ?"
        params = [memory_type]

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if project_path:
            query += " AND project_path = ?"
            params.append(project_path)

        query += " ORDER BY importance DESC, created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [
            {
                "id": row["id"],
                "type": row["type"],
                "content": row["content"],
                "project_path": row["project_path"],
                "session_id": row["session_id"],
                "importance": row["importance"],
                "created_at": row["created_at"]
            }
            for row in rows
        ]

    async def delete_memory(self, memory_id: int) -> bool:
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()
        return cursor.rowcount > 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) as total FROM memories")
        total = cursor.fetchone()["total"]

        cursor.execute("SELECT type, COUNT(*) as count FROM memories GROUP BY type")
        by_type = {row["type"]: row["count"] for row in cursor.fetchall()}

        cursor.execute("SELECT project_path, COUNT(*) as count FROM memories WHERE project_path IS NOT NULL GROUP BY project_path")
        by_project = {row["project_path"]: row["count"] for row in cursor.fetchall()}

        cursor.execute("SELECT agent_type, COUNT(*) as count FROM memories WHERE agent_type IS NOT NULL GROUP BY agent_type")
        by_agent = {row["agent_type"]: row["count"] for row in cursor.fetchall()}

        cursor.execute("SELECT COUNT(*) as count FROM patterns")
        patterns_count = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM projects")
        projects_count = cursor.fetchone()["count"]

        return {
            "total_memories": total,
            "by_type": by_type,
            "by_project": by_project,
            "by_agent": by_agent,
            "patterns_count": patterns_count,
            "projects_count": projects_count,
            "database": self.db_path
        }

    # ============================================================
    # TIMELINE METHODS
    # ============================================================

    async def get_next_sequence_num(self, session_id: str) -> int:
        """Get the next sequence number for a session."""
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT MAX(sequence_num) as max_seq FROM timeline_events WHERE session_id = ?",
            (session_id,)
        )
        row = cursor.fetchone()
        return (row["max_seq"] or 0) + 1

    async def store_timeline_event(
        self,
        session_id: str,
        event_type: str,
        summary: str,
        details: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        project_path: Optional[str] = None,
        parent_event_id: Optional[int] = None,
        root_event_id: Optional[int] = None,
        entities: Optional[Dict[str, List[str]]] = None,
        status: str = "completed",
        outcome: Optional[str] = None,
        confidence: Optional[float] = None,
        is_anchor: bool = False
    ) -> int:
        """Store a timeline event."""
        # Normalize project path to prevent duplicates
        project_path = normalize_path(project_path)

        cursor = self.conn.cursor()

        # Get next sequence number
        sequence_num = await self.get_next_sequence_num(session_id)

        cursor.execute(
            """
            INSERT INTO timeline_events (
                session_id, project_path, event_type, sequence_num,
                summary, details, embedding,
                parent_event_id, root_event_id, entities,
                status, outcome, confidence, is_anchor
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                project_path,
                event_type,
                sequence_num,
                summary[:200] if summary else "",  # Limit summary length
                details,
                self._serialize_embedding(embedding) if embedding else None,
                parent_event_id,
                root_event_id,
                json.dumps(entities) if entities else None,
                status,
                outcome,
                confidence,
                1 if is_anchor else 0
            )
        )
        self.conn.commit()

        # Update session state events counter
        await self._increment_events_since_checkpoint(session_id)

        return cursor.lastrowid

    async def get_timeline_events(
        self,
        session_id: str,
        limit: int = 20,
        event_type: Optional[str] = None,
        since_event_id: Optional[int] = None,
        anchors_only: bool = False
    ) -> List[Dict[str, Any]]:
        """Get timeline events for a session."""
        cursor = self.conn.cursor()

        query = "SELECT * FROM timeline_events WHERE session_id = ?"
        params = [session_id]

        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)

        if since_event_id:
            query += " AND id > ?"
            params.append(since_event_id)

        if anchors_only:
            query += " AND is_anchor = 1"

        query += " ORDER BY sequence_num DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "event_type": row["event_type"],
                "sequence_num": row["sequence_num"],
                "summary": row["summary"],
                "details": row["details"],
                "parent_event_id": row["parent_event_id"],
                "root_event_id": row["root_event_id"],
                "entities": json.loads(row["entities"]) if row["entities"] else None,
                "status": row["status"],
                "outcome": row["outcome"],
                "confidence": row["confidence"],
                "is_anchor": bool(row["is_anchor"]),
                "created_at": row["created_at"]
            }
            for row in rows
        ]

    async def search_timeline_events(
        self,
        embedding: List[float],
        session_id: Optional[str] = None,
        limit: int = 10,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Semantic search across timeline events."""
        cursor = self.conn.cursor()

        query = "SELECT * FROM timeline_events WHERE embedding IS NOT NULL"
        params = []

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            stored_embedding = self._deserialize_embedding(row["embedding"])
            if stored_embedding:
                similarity = self._cosine_similarity(embedding, stored_embedding)
                if similarity >= threshold:
                    results.append({
                        "id": row["id"],
                        "session_id": row["session_id"],
                        "event_type": row["event_type"],
                        "sequence_num": row["sequence_num"],
                        "summary": row["summary"],
                        "details": row["details"],
                        "similarity": similarity,
                        "is_anchor": bool(row["is_anchor"]),
                        "created_at": row["created_at"]
                    })

        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:limit]

    # ============================================================
    # SESSION STATE METHODS
    # ============================================================

    async def get_or_create_session_state(
        self,
        session_id: str,
        project_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get or create session state."""
        # Normalize project path to prevent duplicates
        project_path = normalize_path(project_path)

        cursor = self.conn.cursor()

        cursor.execute("SELECT * FROM session_state WHERE session_id = ?", (session_id,))
        row = cursor.fetchone()

        if row:
            return {
                "id": row["id"],
                "session_id": row["session_id"],
                "project_path": row["project_path"],
                "current_goal": row["current_goal"],
                "pending_questions": json.loads(row["pending_questions"]) if row["pending_questions"] else [],
                "entity_registry": json.loads(row["entity_registry"]) if row["entity_registry"] else {},
                "decisions_summary": row["decisions_summary"],
                "last_checkpoint_id": row["last_checkpoint_id"],
                "events_since_checkpoint": row["events_since_checkpoint"],
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
                "last_activity_at": row["last_activity_at"]
            }

        # Create new session state
        cursor.execute(
            """
            INSERT INTO session_state (session_id, project_path)
            VALUES (?, ?)
            """,
            (session_id, project_path)
        )
        self.conn.commit()

        return {
            "id": cursor.lastrowid,
            "session_id": session_id,
            "project_path": project_path,
            "current_goal": None,
            "pending_questions": [],
            "entity_registry": {},
            "decisions_summary": None,
            "last_checkpoint_id": None,
            "events_since_checkpoint": 0,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "last_activity_at": datetime.now().isoformat()
        }

    async def update_session_state(
        self,
        session_id: str,
        current_goal: Optional[str] = None,
        pending_questions: Optional[List[str]] = None,
        entity_registry: Optional[Dict[str, str]] = None,
        decisions_summary: Optional[str] = None,
        last_checkpoint_id: Optional[int] = None,
        reset_events_counter: bool = False
    ) -> bool:
        """Update session state fields."""
        cursor = self.conn.cursor()

        # Build dynamic update
        updates = ["updated_at = datetime('now')", "last_activity_at = datetime('now')"]
        params = []

        if current_goal is not None:
            updates.append("current_goal = ?")
            params.append(current_goal)

        if pending_questions is not None:
            updates.append("pending_questions = ?")
            params.append(json.dumps(pending_questions))

        if entity_registry is not None:
            updates.append("entity_registry = ?")
            params.append(json.dumps(entity_registry))

        if decisions_summary is not None:
            updates.append("decisions_summary = ?")
            params.append(decisions_summary)

        if last_checkpoint_id is not None:
            updates.append("last_checkpoint_id = ?")
            params.append(last_checkpoint_id)

        if reset_events_counter:
            updates.append("events_since_checkpoint = 0")

        params.append(session_id)

        cursor.execute(
            f"UPDATE session_state SET {', '.join(updates)} WHERE session_id = ?",
            params
        )
        self.conn.commit()
        return cursor.rowcount > 0

    async def _increment_events_since_checkpoint(self, session_id: str):
        """Increment the events counter for a session."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            UPDATE session_state
            SET events_since_checkpoint = events_since_checkpoint + 1,
                last_activity_at = datetime('now')
            WHERE session_id = ?
            """,
            (session_id,)
        )
        self.conn.commit()

    async def get_latest_session_for_project(
        self,
        project_path: str
    ) -> Optional[Dict[str, Any]]:
        """Get the most recent session state for a project."""
        # Normalize project path for consistent matching
        project_path = normalize_path(project_path)

        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM session_state
            WHERE project_path = ?
            ORDER BY last_activity_at DESC
            LIMIT 1
            """,
            (project_path,)
        )
        row = cursor.fetchone()

        if row:
            return {
                "id": row["id"],
                "session_id": row["session_id"],
                "project_path": row["project_path"],
                "current_goal": row["current_goal"],
                "pending_questions": json.loads(row["pending_questions"]) if row["pending_questions"] else [],
                "entity_registry": json.loads(row["entity_registry"]) if row["entity_registry"] else {},
                "decisions_summary": row["decisions_summary"],
                "last_checkpoint_id": row["last_checkpoint_id"],
                "events_since_checkpoint": row["events_since_checkpoint"],
                "last_activity_at": row["last_activity_at"]
            }
        return None

    # ============================================================
    # CHECKPOINT METHODS
    # ============================================================

    async def store_checkpoint(
        self,
        session_id: str,
        summary: str,
        event_id: Optional[int] = None,
        key_facts: Optional[List[str]] = None,
        decisions: Optional[List[str]] = None,
        entities: Optional[Dict[str, str]] = None,
        current_goal: Optional[str] = None,
        pending_items: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
        event_count: Optional[int] = None
    ) -> int:
        """Store a checkpoint."""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO checkpoints (
                session_id, event_id, summary, key_facts, decisions,
                entities, current_goal, pending_items, embedding, event_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                event_id,
                summary,
                json.dumps(key_facts) if key_facts else None,
                json.dumps(decisions) if decisions else None,
                json.dumps(entities) if entities else None,
                current_goal,
                json.dumps(pending_items) if pending_items else None,
                self._serialize_embedding(embedding) if embedding else None,
                event_count
            )
        )
        self.conn.commit()

        checkpoint_id = cursor.lastrowid

        # Update session state with new checkpoint
        await self.update_session_state(
            session_id,
            last_checkpoint_id=checkpoint_id,
            reset_events_counter=True
        )

        return checkpoint_id

    async def get_latest_checkpoint(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get the latest checkpoint for a session."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM checkpoints
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (session_id,)
        )
        row = cursor.fetchone()

        if row:
            return {
                "id": row["id"],
                "session_id": row["session_id"],
                "event_id": row["event_id"],
                "summary": row["summary"],
                "key_facts": json.loads(row["key_facts"]) if row["key_facts"] else [],
                "decisions": json.loads(row["decisions"]) if row["decisions"] else [],
                "entities": json.loads(row["entities"]) if row["entities"] else {},
                "current_goal": row["current_goal"],
                "pending_items": json.loads(row["pending_items"]) if row["pending_items"] else [],
                "event_count": row["event_count"],
                "created_at": row["created_at"]
            }
        return None

    async def get_checkpoints_for_session(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get all checkpoints for a session."""
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT * FROM checkpoints
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_id, limit)
        )

        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "summary": row["summary"],
                "current_goal": row["current_goal"],
                "event_count": row["event_count"],
                "created_at": row["created_at"]
            }
            for row in cursor.fetchall()
        ]

    # ============================================================
    # GENERIC QUERY METHOD
    # ============================================================

    @with_retry(max_retries=DB_MAX_RETRIES, base_delay=DB_RETRY_BASE_DELAY)
    async def execute_query(
        self,
        query: str,
        params: tuple = (),
        timeout: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Execute a raw SQL query and return results as list of dicts.

        Args:
            query: SQL query to execute
            params: Query parameters
            timeout: Optional query timeout in seconds (uses DB_TIMEOUT if not specified)

        Returns:
            List of dictionaries representing rows

        Raises:
            QueryTimeoutError: If query exceeds timeout
            RetryExhaustedError: If all retry attempts fail
            DatabaseError: For other database errors
        """
        effective_timeout = timeout or DB_TIMEOUT

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()

                # Set timeout for this query
                start_time = time.time()

                cursor.execute(query, params)
                rows = cursor.fetchall()

                # Check if query took too long (for logging/monitoring)
                elapsed = time.time() - start_time
                if elapsed > effective_timeout * 0.8:
                    logger.warning(
                        f"Slow query detected ({elapsed:.2f}s): {query[:100]}..."
                    )

                if not rows:
                    return []

                # Convert Row objects to dicts
                return [dict(row) for row in rows]

        except sqlite3.OperationalError as e:
            error_str = str(e).lower()
            if "database is locked" in error_str or "busy" in error_str:
                logger.warning(f"Database busy/locked, will retry: {e}")
                raise  # Let retry decorator handle it
            elif "unable to open database" in error_str:
                raise ConnectionPoolError(f"Cannot open database: {e}", original_error=e)
            else:
                raise DatabaseError(
                    f"Query execution failed: {e}",
                    error_code="DB_QUERY_ERROR",
                    original_error=e
                )
        except sqlite3.IntegrityError as e:
            raise DatabaseError(
                f"Integrity constraint violation: {e}",
                error_code="DB_INTEGRITY_ERROR",
                original_error=e
            )
        except Exception as e:
            logger.error(f"Unexpected error executing query: {e}")
            raise DatabaseError(
                f"Unexpected database error: {e}",
                error_code="DB_UNKNOWN_ERROR",
                original_error=e
            )

    async def execute_write(
        self,
        query: str,
        params: tuple = (),
        commit: bool = True
    ) -> int:
        """Execute a write query (INSERT, UPDATE, DELETE) with retry logic.

        Args:
            query: SQL query to execute
            params: Query parameters
            commit: Whether to commit the transaction

        Returns:
            Number of affected rows (or lastrowid for INSERT)

        Raises:
            RetryExhaustedError: If all retry attempts fail
            DatabaseError: For other database errors
        """
        return await self._execute_write_with_retry(query, params, commit)

    @with_retry(max_retries=DB_MAX_RETRIES, base_delay=DB_RETRY_BASE_DELAY)
    async def _execute_write_with_retry(
        self,
        query: str,
        params: tuple,
        commit: bool
    ) -> int:
        """Internal write execution with retry decorator."""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)

                if commit:
                    conn.commit()

                # Return lastrowid for INSERT, rowcount for UPDATE/DELETE
                if query.strip().upper().startswith("INSERT"):
                    return cursor.lastrowid
                return cursor.rowcount

        except sqlite3.OperationalError as e:
            error_str = str(e).lower()
            if "database is locked" in error_str or "busy" in error_str:
                logger.warning(f"Database busy/locked during write, will retry: {e}")
                raise  # Let retry decorator handle it
            raise DatabaseError(
                f"Write operation failed: {e}",
                error_code="DB_WRITE_ERROR",
                original_error=e
            )
        except sqlite3.IntegrityError as e:
            raise DatabaseError(
                f"Integrity constraint violation: {e}",
                error_code="DB_INTEGRITY_ERROR",
                original_error=e
            )

    # ============================================================
    # KNOWLEDGE GRAPH RELATIONSHIP METHODS (Internal)
    # ============================================================

    async def create_relationship(
        self,
        source_id: int,
        target_id: int,
        relationship: str,
        strength: float = 1.0
    ) -> dict:
        """Create a relationship between two memories.

        Relationship types:
            - fixes: source memory fixes the issue in target
            - caused_by: source issue was caused by target
            - supports: source evidence supports target conclusion
            - contradicts: source contradicts target
            - related: general semantic relationship
            - follows: source chronologically follows target

        Args:
            source_id: ID of the source memory
            target_id: ID of the target memory
            relationship: Type of relationship
            strength: Relationship strength (0.0 to 1.0)

        Returns:
            Dict with relationship details or error
        """
        valid_relationships = {'fixes', 'caused_by', 'supports', 'contradicts', 'related', 'follows'}
        if relationship not in valid_relationships:
            return {
                "success": False,
                "error": f"Invalid relationship type. Must be one of: {valid_relationships}"
            }

        # Verify both memories exist
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM memories WHERE id IN (?, ?)", (source_id, target_id))
        existing = {row["id"] for row in cursor.fetchall()}

        if source_id not in existing:
            return {"success": False, "error": f"Source memory {source_id} not found"}
        if target_id not in existing:
            return {"success": False, "error": f"Target memory {target_id} not found"}

        try:
            cursor.execute("""
                INSERT INTO memory_relationships (source_id, target_id, relationship, strength)
                VALUES (?, ?, ?, ?)
            """, (source_id, target_id, relationship, strength))
            self.conn.commit()

            return {
                "success": True,
                "id": cursor.lastrowid,
                "source_id": source_id,
                "target_id": target_id,
                "relationship": relationship,
                "strength": strength
            }
        except sqlite3.IntegrityError:
            # Relationship already exists, update strength
            cursor.execute("""
                UPDATE memory_relationships
                SET strength = ?, created_at = CURRENT_TIMESTAMP
                WHERE source_id = ? AND target_id = ? AND relationship = ?
            """, (strength, source_id, target_id, relationship))
            self.conn.commit()

            return {
                "success": True,
                "updated": True,
                "source_id": source_id,
                "target_id": target_id,
                "relationship": relationship,
                "strength": strength
            }

    async def get_related_memories(
        self,
        memory_id: int,
        relationship: str = None,
        direction: str = 'both',
        depth: int = 1
    ) -> list:
        """Get memories related to this one.

        Args:
            memory_id: ID of the memory to find relationships for
            relationship: Optional filter by relationship type
            direction: 'outgoing' (this->other), 'incoming' (other->this), 'both'
            depth: How many levels deep to traverse (1 = direct only)

        Returns:
            List of related memories with relationship info
        """
        cursor = self.conn.cursor()
        results = []
        visited = {memory_id}

        async def traverse(current_id: int, current_depth: int):
            if current_depth > depth:
                return

            # Build query based on direction
            queries = []
            if direction in ('outgoing', 'both'):
                q = """
                    SELECT mr.target_id as related_id, mr.relationship, mr.strength, 'outgoing' as direction,
                           m.type, m.content, m.project_path, m.importance, m.created_at
                    FROM memory_relationships mr
                    JOIN memories m ON m.id = mr.target_id
                    WHERE mr.source_id = ?
                """
                params = [current_id]
                if relationship:
                    q += " AND mr.relationship = ?"
                    params.append(relationship)
                queries.append((q, params))

            if direction in ('incoming', 'both'):
                q = """
                    SELECT mr.source_id as related_id, mr.relationship, mr.strength, 'incoming' as direction,
                           m.type, m.content, m.project_path, m.importance, m.created_at
                    FROM memory_relationships mr
                    JOIN memories m ON m.id = mr.source_id
                    WHERE mr.target_id = ?
                """
                params = [current_id]
                if relationship:
                    q += " AND mr.relationship = ?"
                    params.append(relationship)
                queries.append((q, params))

            for query, params in queries:
                cursor.execute(query, params)
                for row in cursor.fetchall():
                    related_id = row["related_id"]
                    if related_id not in visited:
                        visited.add(related_id)
                        results.append({
                            "id": related_id,
                            "relationship": row["relationship"],
                            "strength": row["strength"],
                            "direction": row["direction"],
                            "depth": current_depth,
                            "type": row["type"],
                            "content": row["content"][:200] + "..." if len(row["content"]) > 200 else row["content"],
                            "project_path": row["project_path"],
                            "importance": row["importance"],
                            "created_at": row["created_at"]
                        })

                        # Recurse for deeper traversal
                        if current_depth < depth:
                            await traverse(related_id, current_depth + 1)

        await traverse(memory_id, 1)
        return results

    async def get_causal_chain(self, memory_id: int, max_depth: int = 5) -> dict:
        """Traverse the fixes/caused_by chain to find root cause and all fixes.

        Args:
            memory_id: Starting memory ID
            max_depth: Maximum traversal depth to prevent infinite loops

        Returns:
            Dict with root_causes, fixes, and the full chain
        """
        cursor = self.conn.cursor()

        root_causes = []
        fixes = []
        chain = []
        visited = {memory_id}

        # Traverse backwards to find root causes (caused_by)
        async def find_causes(current_id: int, depth: int):
            if depth > max_depth:
                return

            cursor.execute("""
                SELECT mr.target_id, m.type, m.content, m.project_path, m.created_at
                FROM memory_relationships mr
                JOIN memories m ON m.id = mr.target_id
                WHERE mr.source_id = ? AND mr.relationship = 'caused_by'
            """, (current_id,))

            rows = cursor.fetchall()
            if not rows:
                # No more causes, this is a root cause
                cursor.execute("SELECT * FROM memories WHERE id = ?", (current_id,))
                row = cursor.fetchone()
                if row and current_id != memory_id:
                    root_causes.append({
                        "id": current_id,
                        "type": row["type"],
                        "content": row["content"][:200] + "..." if len(row["content"]) > 200 else row["content"],
                        "project_path": row["project_path"],
                        "created_at": row["created_at"]
                    })
            else:
                for row in rows:
                    target_id = row["target_id"]
                    if target_id not in visited:
                        visited.add(target_id)
                        chain.append({
                            "from": current_id,
                            "to": target_id,
                            "relationship": "caused_by"
                        })
                        await find_causes(target_id, depth + 1)

        # Traverse forwards to find fixes
        async def find_fixes(current_id: int, depth: int):
            if depth > max_depth:
                return

            cursor.execute("""
                SELECT mr.source_id, m.type, m.content, m.project_path, m.created_at, m.success
                FROM memory_relationships mr
                JOIN memories m ON m.id = mr.source_id
                WHERE mr.target_id = ? AND mr.relationship = 'fixes'
            """, (current_id,))

            for row in cursor.fetchall():
                source_id = row["source_id"]
                if source_id not in visited:
                    visited.add(source_id)
                    fixes.append({
                        "id": source_id,
                        "type": row["type"],
                        "content": row["content"][:200] + "..." if len(row["content"]) > 200 else row["content"],
                        "project_path": row["project_path"],
                        "created_at": row["created_at"],
                        "success": bool(row["success"]) if row["success"] is not None else None
                    })
                    chain.append({
                        "from": source_id,
                        "to": current_id,
                        "relationship": "fixes"
                    })
                    await find_fixes(source_id, depth + 1)

        await find_causes(memory_id, 1)
        # Reset visited for fix traversal but keep memory_id
        visited = {memory_id}
        await find_fixes(memory_id, 1)

        return {
            "memory_id": memory_id,
            "root_causes": root_causes,
            "fixes": fixes,
            "chain": chain,
            "total_related": len(root_causes) + len(fixes)
        }

    async def find_contradictions(self, memory_id: int) -> list:
        """Find memories that contradict this one.

        Args:
            memory_id: Memory to find contradictions for

        Returns:
            List of contradicting memories
        """
        cursor = self.conn.cursor()

        # Get contradictions in both directions
        cursor.execute("""
            SELECT m.id, m.type, m.content, m.project_path, m.importance, m.created_at,
                   mr.strength, 'outgoing' as direction
            FROM memory_relationships mr
            JOIN memories m ON m.id = mr.target_id
            WHERE mr.source_id = ? AND mr.relationship = 'contradicts'
            UNION ALL
            SELECT m.id, m.type, m.content, m.project_path, m.importance, m.created_at,
                   mr.strength, 'incoming' as direction
            FROM memory_relationships mr
            JOIN memories m ON m.id = mr.source_id
            WHERE mr.target_id = ? AND mr.relationship = 'contradicts'
        """, (memory_id, memory_id))

        contradictions = []
        seen = set()
        for row in cursor.fetchall():
            if row["id"] not in seen:
                seen.add(row["id"])
                contradictions.append({
                    "id": row["id"],
                    "type": row["type"],
                    "content": row["content"][:200] + "..." if len(row["content"]) > 200 else row["content"],
                    "project_path": row["project_path"],
                    "importance": row["importance"],
                    "created_at": row["created_at"],
                    "contradiction_strength": row["strength"]
                })

        return contradictions

    async def get_graph_data(self, project_path: str = None, limit: int = 200) -> dict:
        """Get nodes and edges for graph visualization.

        Args:
            project_path: Optional filter by project
            limit: Maximum number of nodes to return

        Returns:
            Dict with nodes, edges, and stats for visualization
        """
        project_path = normalize_path(project_path)
        cursor = self.conn.cursor()

        # Get nodes (memories)
        if project_path:
            cursor.execute("""
                SELECT id, type, content, project_path, importance, created_at
                FROM memories
                WHERE project_path = ?
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
            """, (project_path, limit))
        else:
            cursor.execute("""
                SELECT id, type, content, project_path, importance, created_at
                FROM memories
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
            """, (limit,))

        nodes = []
        node_ids = set()
        for row in cursor.fetchall():
            node_ids.add(row["id"])
            nodes.append({
                "id": row["id"],
                "type": row["type"],
                "content": row["content"][:100] + "..." if len(row["content"]) > 100 else row["content"],
                "project_path": row["project_path"],
                "importance": row["importance"],
                "created_at": row["created_at"]
            })

        # Get edges (relationships between these nodes)
        if node_ids:
            placeholders = ",".join("?" * len(node_ids))
            cursor.execute(f"""
                SELECT source_id, target_id, relationship, strength
                FROM memory_relationships
                WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders})
            """, list(node_ids) + list(node_ids))

            edges = [
                {
                    "source": row["source_id"],
                    "target": row["target_id"],
                    "relationship": row["relationship"],
                    "strength": row["strength"]
                }
                for row in cursor.fetchall()
            ]
        else:
            edges = []

        # Get stats
        cursor.execute("SELECT COUNT(*) as total FROM memory_relationships")
        total_relationships = cursor.fetchone()["total"]

        cursor.execute("""
            SELECT relationship, COUNT(*) as count
            FROM memory_relationships
            GROUP BY relationship
        """)
        relationship_counts = {row["relationship"]: row["count"] for row in cursor.fetchall()}

        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "total_relationships_in_db": total_relationships,
                "relationship_counts": relationship_counts
            }
        }

    async def get_subgraph(self, memory_id: int, depth: int = 2) -> dict:
        """Get connected subgraph from a starting node.

        Args:
            memory_id: Starting node ID
            depth: How many hops to traverse

        Returns:
            Dict with nodes and edges in the subgraph
        """
        cursor = self.conn.cursor()

        # Collect all connected node IDs
        node_ids = {memory_id}
        current_frontier = {memory_id}

        for _ in range(depth):
            if not current_frontier:
                break

            placeholders = ",".join("?" * len(current_frontier))

            # Get connected nodes
            cursor.execute(f"""
                SELECT DISTINCT target_id as connected_id FROM memory_relationships
                WHERE source_id IN ({placeholders})
                UNION
                SELECT DISTINCT source_id as connected_id FROM memory_relationships
                WHERE target_id IN ({placeholders})
            """, list(current_frontier) + list(current_frontier))

            new_nodes = {row["connected_id"] for row in cursor.fetchall()}
            current_frontier = new_nodes - node_ids
            node_ids.update(new_nodes)

        # Get node details
        nodes = []
        if node_ids:
            placeholders = ",".join("?" * len(node_ids))
            cursor.execute(f"""
                SELECT id, type, content, project_path, importance, created_at
                FROM memories
                WHERE id IN ({placeholders})
            """, list(node_ids))

            for row in cursor.fetchall():
                nodes.append({
                    "id": row["id"],
                    "type": row["type"],
                    "content": row["content"][:100] + "..." if len(row["content"]) > 100 else row["content"],
                    "project_path": row["project_path"],
                    "importance": row["importance"],
                    "created_at": row["created_at"],
                    "is_center": row["id"] == memory_id
                })

            # Get edges between these nodes
            cursor.execute(f"""
                SELECT source_id, target_id, relationship, strength
                FROM memory_relationships
                WHERE source_id IN ({placeholders}) AND target_id IN ({placeholders})
            """, list(node_ids) + list(node_ids))

            edges = [
                {
                    "source": row["source_id"],
                    "target": row["target_id"],
                    "relationship": row["relationship"],
                    "strength": row["strength"]
                }
                for row in cursor.fetchall()
            ]
        else:
            edges = []

        return {
            "center_id": memory_id,
            "depth": depth,
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges)
            }
        }

    async def get_relationship_stats(self, project_path: str = None) -> dict:
        """Get counts of each relationship type.

        Args:
            project_path: Optional filter by project

        Returns:
            Dict with relationship statistics
        """
        project_path = normalize_path(project_path)
        cursor = self.conn.cursor()

        if project_path:
            cursor.execute("""
                SELECT mr.relationship, COUNT(*) as count
                FROM memory_relationships mr
                JOIN memories m ON m.id = mr.source_id
                WHERE m.project_path = ?
                GROUP BY mr.relationship
                ORDER BY count DESC
            """, (project_path,))
        else:
            cursor.execute("""
                SELECT relationship, COUNT(*) as count
                FROM memory_relationships
                GROUP BY relationship
                ORDER BY count DESC
            """)

        by_type = {row["relationship"]: row["count"] for row in cursor.fetchall()}

        # Get total
        total = sum(by_type.values())

        # Get most connected memories
        cursor.execute("""
            SELECT m.id, m.type, m.content, COUNT(*) as connection_count
            FROM memories m
            JOIN (
                SELECT source_id as memory_id FROM memory_relationships
                UNION ALL
                SELECT target_id as memory_id FROM memory_relationships
            ) r ON r.memory_id = m.id
            GROUP BY m.id
            ORDER BY connection_count DESC
            LIMIT 10
        """)

        most_connected = [
            {
                "id": row["id"],
                "type": row["type"],
                "content": row["content"][:100] + "..." if len(row["content"]) > 100 else row["content"],
                "connection_count": row["connection_count"]
            }
            for row in cursor.fetchall()
        ]

        return {
            "total_relationships": total,
            "by_type": by_type,
            "most_connected_memories": most_connected,
            "project_path": project_path
        }
