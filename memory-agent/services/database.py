"""Database service using SQLite with numpy-based vector similarity."""
import os
import json
import sqlite3
import numpy as np
from datetime import datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv("DATABASE_PATH", str(Path(__file__).parent.parent / "memories.db"))


def normalize_path(path: str) -> str:
    """Normalize file paths to prevent duplicates from different separators.

    Converts all paths to forward slashes (Unix-style) for consistency.
    This prevents 'C:/foo' and 'C:\\foo' being treated as different projects.
    """
    if not path:
        return path
    # Convert to forward slashes and remove trailing slashes
    normalized = path.replace("\\", "/").rstrip("/")
    return normalized


class DatabaseService:
    """Service for vector storage and retrieval using SQLite + numpy."""

    def __init__(self):
        self.db_path = DB_PATH
        self.conn: Optional[sqlite3.Connection] = None

    async def connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    async def disconnect(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

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
        importance: int = 5
    ) -> int:
        """Store a memory with full context."""
        # Normalize project path to prevent duplicates
        project_path = normalize_path(project_path)

        cursor = self.conn.cursor()
        cursor.execute(
            """
            INSERT INTO memories (
                type, content, embedding, metadata,
                project_path, project_name, project_type, tech_stack,
                session_id, chat_id,
                agent_type, skill_used, tools_used,
                outcome, success,
                tags, importance
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                importance
            )
        )
        self.conn.commit()
        return cursor.lastrowid

    async def search_similar(
        self,
        embedding: List[float],
        limit: int = 10,
        memory_type: Optional[str] = None,
        session_id: Optional[str] = None,
        project_path: Optional[str] = None,
        agent_type: Optional[str] = None,
        success_only: bool = False,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Search for similar memories with optional filters."""
        # Normalize project path for consistent matching
        project_path = normalize_path(project_path)

        cursor = self.conn.cursor()

        query = """
            SELECT id, type, content, embedding, metadata,
                   project_path, project_name, project_type, tech_stack,
                   session_id, chat_id, agent_type, skill_used, tools_used,
                   outcome, success, tags, importance, created_at
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
                        "type": row["type"],
                        "content": row["content"],
                        "similarity": similarity,
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
                        "tags": json.loads(row["tags"]) if row["tags"] else None,
                        "importance": row["importance"],
                        "created_at": row["created_at"],
                        "metadata": json.loads(row["metadata"]) if row["metadata"] else {}
                    })

        # Sort by similarity * importance for better ranking
        results.sort(key=lambda x: x["similarity"] * (x["importance"] / 10), reverse=True)

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

    async def update_pattern_outcome(self, pattern_id: int, success: bool):
        """Update pattern success/failure count."""
        cursor = self.conn.cursor()
        if success:
            cursor.execute("UPDATE patterns SET success_count = success_count + 1 WHERE id = ?", (pattern_id,))
        else:
            cursor.execute("UPDATE patterns SET failure_count = failure_count + 1 WHERE id = ?", (pattern_id,))
        self.conn.commit()

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

    async def execute_query(
        self,
        query: str,
        params: tuple = ()
    ) -> List[Dict[str, Any]]:
        """Execute a raw SQL query and return results as list of dicts."""
        cursor = self.conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

        if not rows:
            return []

        # Convert Row objects to dicts
        return [dict(row) for row in rows]
