"""Retry queue for hook failures with persistence and exponential backoff.

Ensures hook calls are not lost when the memory agent is unavailable.
Uses SQLite for persistence and supports file-based fallback.
"""
import os
import json
import time
import sqlite3
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from threading import Lock
from dotenv import load_dotenv

load_dotenv()

# Configuration
try:
    from config import USER_DATA_DIR as _DATA_DIR
except ImportError:
    _DATA_DIR = Path.home() / ".claude-memory"
QUEUE_DB_PATH = os.getenv("QUEUE_DB_PATH", str(_DATA_DIR / "queue.db"))
QUEUE_FILE_FALLBACK = os.getenv("QUEUE_FILE_FALLBACK", str(Path.home() / ".claude" / "memory_queue.jsonl"))
MAX_RETRIES = int(os.getenv("QUEUE_MAX_RETRIES", "5"))
BASE_BACKOFF_SECONDS = float(os.getenv("QUEUE_BASE_BACKOFF", "1.0"))
MAX_BACKOFF_SECONDS = float(os.getenv("QUEUE_MAX_BACKOFF", "300.0"))  # 5 minutes max


class RetryQueue:
    """SQLite-backed retry queue with exponential backoff.

    Features:
    - Persistent storage in SQLite
    - File-based fallback when SQLite unavailable
    - Exponential backoff for retries
    - Dead letter queue for permanently failed requests
    - Background processing with configurable interval
    """

    def __init__(self, db_path: str = QUEUE_DB_PATH):
        self.db_path = db_path
        self.fallback_path = Path(QUEUE_FILE_FALLBACK)
        self.conn: Optional[sqlite3.Connection] = None
        self._lock = Lock()
        self._processing = False
        self._processor_task: Optional[asyncio.Task] = None

        # Stats
        self._enqueued = 0
        self._processed = 0
        self._failed = 0
        self._retried = 0

        self._initialize_db()

    def _initialize_db(self):
        """Initialize the queue database."""
        try:
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.row_factory = sqlite3.Row
            cursor = self.conn.cursor()

            # Queue table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pending_requests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT NOT NULL,
                    method TEXT DEFAULT 'POST',
                    payload TEXT NOT NULL,
                    headers TEXT,
                    attempts INTEGER DEFAULT 0,
                    max_attempts INTEGER DEFAULT 5,
                    created_at TEXT DEFAULT (datetime('now')),
                    next_retry_at TEXT DEFAULT (datetime('now')),
                    last_error TEXT,
                    status TEXT DEFAULT 'pending'
                )
            """)

            # Dead letter queue for permanently failed requests
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dead_letters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    original_id INTEGER,
                    endpoint TEXT NOT NULL,
                    method TEXT,
                    payload TEXT NOT NULL,
                    headers TEXT,
                    attempts INTEGER,
                    last_error TEXT,
                    created_at TEXT,
                    failed_at TEXT DEFAULT (datetime('now'))
                )
            """)

            # Indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pending_status ON pending_requests(status)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_pending_retry ON pending_requests(next_retry_at)")

            self.conn.commit()
        except Exception as e:
            # Fall back to file-based queue
            self.conn = None
            self._ensure_fallback_dir()

    def _ensure_fallback_dir(self):
        """Ensure the fallback directory exists."""
        self.fallback_path.parent.mkdir(parents=True, exist_ok=True)

    def enqueue(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        method: str = "POST",
        headers: Optional[Dict[str, str]] = None,
        max_attempts: int = MAX_RETRIES
    ) -> int:
        """Add a request to the retry queue.

        Args:
            endpoint: API endpoint URL
            payload: Request payload (will be JSON serialized)
            method: HTTP method
            headers: Optional headers
            max_attempts: Maximum retry attempts

        Returns:
            Queue item ID (or -1 for file fallback)
        """
        with self._lock:
            self._enqueued += 1

            if self.conn:
                try:
                    cursor = self.conn.cursor()
                    cursor.execute(
                        """
                        INSERT INTO pending_requests
                        (endpoint, method, payload, headers, max_attempts)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            endpoint,
                            method,
                            json.dumps(payload),
                            json.dumps(headers) if headers else None,
                            max_attempts
                        )
                    )
                    self.conn.commit()
                    return cursor.lastrowid
                except Exception:
                    pass  # Fall through to file fallback

            # File-based fallback
            self._ensure_fallback_dir()
            item = {
                "endpoint": endpoint,
                "method": method,
                "payload": payload,
                "headers": headers,
                "attempts": 0,
                "max_attempts": max_attempts,
                "created_at": datetime.now().isoformat(),
                "status": "pending"
            }
            with open(self.fallback_path, "a") as f:
                f.write(json.dumps(item) + "\n")
            return -1

    def get_pending(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get pending requests ready for retry.

        Args:
            limit: Maximum number of items to return

        Returns:
            List of pending request dictionaries
        """
        with self._lock:
            if self.conn:
                try:
                    cursor = self.conn.cursor()
                    cursor.execute(
                        """
                        SELECT * FROM pending_requests
                        WHERE status = 'pending'
                        AND datetime(next_retry_at) <= datetime('now')
                        ORDER BY next_retry_at ASC
                        LIMIT ?
                        """,
                        (limit,)
                    )
                    rows = cursor.fetchall()
                    return [dict(row) for row in rows]
                except Exception:
                    pass

            # File fallback
            if self.fallback_path.exists():
                items = []
                with open(self.fallback_path, "r") as f:
                    for line in f:
                        try:
                            item = json.loads(line.strip())
                            if item.get("status") == "pending":
                                items.append(item)
                                if len(items) >= limit:
                                    break
                        except json.JSONDecodeError:
                            continue
                return items

            return []

    def mark_success(self, item_id: int):
        """Mark a request as successfully processed."""
        with self._lock:
            self._processed += 1
            if self.conn and item_id > 0:
                try:
                    cursor = self.conn.cursor()
                    cursor.execute(
                        "DELETE FROM pending_requests WHERE id = ?",
                        (item_id,)
                    )
                    self.conn.commit()
                except Exception:
                    pass

    def mark_failed(self, item_id: int, error: str):
        """Mark a request as failed and schedule retry or move to dead letter queue."""
        with self._lock:
            self._retried += 1

            if self.conn and item_id > 0:
                try:
                    cursor = self.conn.cursor()

                    # Get current item
                    cursor.execute("SELECT * FROM pending_requests WHERE id = ?", (item_id,))
                    row = cursor.fetchone()
                    if not row:
                        return

                    attempts = row["attempts"] + 1
                    max_attempts = row["max_attempts"]

                    if attempts >= max_attempts:
                        # Move to dead letter queue
                        self._failed += 1
                        cursor.execute(
                            """
                            INSERT INTO dead_letters
                            (original_id, endpoint, method, payload, headers, attempts, last_error, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                item_id,
                                row["endpoint"],
                                row["method"],
                                row["payload"],
                                row["headers"],
                                attempts,
                                error,
                                row["created_at"]
                            )
                        )
                        cursor.execute("DELETE FROM pending_requests WHERE id = ?", (item_id,))
                    else:
                        # Calculate next retry with exponential backoff
                        backoff = min(
                            BASE_BACKOFF_SECONDS * (2 ** attempts),
                            MAX_BACKOFF_SECONDS
                        )
                        next_retry = datetime.now() + timedelta(seconds=backoff)

                        cursor.execute(
                            """
                            UPDATE pending_requests
                            SET attempts = ?, last_error = ?, next_retry_at = ?, status = 'pending'
                            WHERE id = ?
                            """,
                            (attempts, error, next_retry.isoformat(), item_id)
                        )

                    self.conn.commit()
                except Exception:
                    pass

    def get_queue_depth(self) -> int:
        """Get the number of pending requests."""
        with self._lock:
            if self.conn:
                try:
                    cursor = self.conn.cursor()
                    cursor.execute("SELECT COUNT(*) as count FROM pending_requests WHERE status = 'pending'")
                    row = cursor.fetchone()
                    return row["count"] if row else 0
                except Exception:
                    pass

            # File fallback
            if self.fallback_path.exists():
                count = 0
                with open(self.fallback_path, "r") as f:
                    for line in f:
                        try:
                            item = json.loads(line.strip())
                            if item.get("status") == "pending":
                                count += 1
                        except json.JSONDecodeError:
                            continue
                return count

            return 0

    def get_dead_letters(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get items from the dead letter queue."""
        with self._lock:
            if self.conn:
                try:
                    cursor = self.conn.cursor()
                    cursor.execute(
                        "SELECT * FROM dead_letters ORDER BY failed_at DESC LIMIT ?",
                        (limit,)
                    )
                    rows = cursor.fetchall()
                    return [dict(row) for row in rows]
                except Exception:
                    pass
            return []

    def retry_dead_letter(self, dead_letter_id: int) -> bool:
        """Move a dead letter back to the pending queue."""
        with self._lock:
            if self.conn:
                try:
                    cursor = self.conn.cursor()
                    cursor.execute("SELECT * FROM dead_letters WHERE id = ?", (dead_letter_id,))
                    row = cursor.fetchone()
                    if not row:
                        return False

                    cursor.execute(
                        """
                        INSERT INTO pending_requests
                        (endpoint, method, payload, headers, attempts, max_attempts, created_at)
                        VALUES (?, ?, ?, ?, 0, ?, ?)
                        """,
                        (
                            row["endpoint"],
                            row["method"],
                            row["payload"],
                            row["headers"],
                            MAX_RETRIES,
                            row["created_at"]
                        )
                    )
                    cursor.execute("DELETE FROM dead_letters WHERE id = ?", (dead_letter_id,))
                    self.conn.commit()
                    return True
                except Exception:
                    pass
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        dead_letter_count = 0
        if self.conn:
            try:
                cursor = self.conn.cursor()
                cursor.execute("SELECT COUNT(*) as count FROM dead_letters")
                row = cursor.fetchone()
                dead_letter_count = row["count"] if row else 0
            except Exception:
                pass

        return {
            "queue_depth": self.get_queue_depth(),
            "dead_letters": dead_letter_count,
            "total_enqueued": self._enqueued,
            "total_processed": self._processed,
            "total_failed": self._failed,
            "total_retried": self._retried,
            "db_path": self.db_path,
            "fallback_path": str(self.fallback_path),
            "using_db": self.conn is not None
        }

    async def process_queue(
        self,
        processor: Callable[[Dict[str, Any]], bool],
        batch_size: int = 10,
        interval_seconds: float = 5.0
    ):
        """Background task to process the queue.

        Args:
            processor: Async function that processes a single item, returns True on success
            batch_size: Number of items to process per batch
            interval_seconds: Time between processing batches
        """
        self._processing = True

        while self._processing:
            try:
                items = self.get_pending(limit=batch_size)

                for item in items:
                    try:
                        success = await processor(item)
                        if success:
                            self.mark_success(item.get("id", -1))
                        else:
                            self.mark_failed(item.get("id", -1), "Processor returned False")
                    except Exception as e:
                        self.mark_failed(item.get("id", -1), str(e))

            except Exception:
                pass  # Don't crash the background task

            await asyncio.sleep(interval_seconds)

    def stop_processing(self):
        """Stop the background processing task."""
        self._processing = False

    def close(self):
        """Close the database connection."""
        self.stop_processing()
        if self.conn:
            self.conn.close()


# Global instance
_queue: Optional[RetryQueue] = None


def get_queue() -> RetryQueue:
    """Get the global retry queue instance."""
    global _queue
    if _queue is None:
        _queue = RetryQueue()
    return _queue


async def enqueue_request(
    endpoint: str,
    payload: Dict[str, Any],
    method: str = "POST",
    headers: Optional[Dict[str, str]] = None
) -> int:
    """Convenience function to enqueue a request."""
    return get_queue().enqueue(endpoint, payload, method, headers)
