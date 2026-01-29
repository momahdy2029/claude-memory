"""Memory cleanup and pruning service.

Handles automatic cleanup of old, low-value, and duplicate memories.
Supports archival before deletion and configurable retention policies.
"""
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict


class CleanupService:
    """Service for memory cleanup, deduplication, and archival.

    Features:
    - Relevance-based cleanup (low-scoring memories)
    - Age-based retention (older than N days)
    - Duplicate detection and merging
    - Soft-delete with archive for recovery
    - Per-project configuration
    - Dry-run mode for preview
    - Audit logging
    """

    def __init__(self, db, embeddings=None):
        self.db = db
        self.embeddings = embeddings

    async def get_config(
        self,
        project_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get cleanup configuration for a project or global default."""
        cursor = self.db.conn.cursor()

        if project_path:
            cursor.execute(
                "SELECT * FROM cleanup_config WHERE project_path = ?",
                (project_path,)
            )
            row = cursor.fetchone()
            if row:
                return dict(row)

        # Return defaults
        return {
            "retention_days": 90,
            "min_relevance_score": 0.1,
            "keep_high_importance": True,
            "importance_threshold": 7,
            "dedup_enabled": True,
            "dedup_threshold": 0.95,
            "archive_before_delete": True,
            "archive_retention_days": 365,
            "auto_cleanup_enabled": False
        }

    async def save_config(
        self,
        project_path: Optional[str],
        config: Dict[str, Any]
    ) -> bool:
        """Save cleanup configuration for a project."""
        cursor = self.db.conn.cursor()

        cursor.execute(
            """
            INSERT INTO cleanup_config (
                project_path, retention_days, min_relevance_score,
                keep_high_importance, importance_threshold,
                dedup_enabled, dedup_threshold,
                archive_before_delete, archive_retention_days,
                auto_cleanup_enabled
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_path) DO UPDATE SET
                retention_days = excluded.retention_days,
                min_relevance_score = excluded.min_relevance_score,
                keep_high_importance = excluded.keep_high_importance,
                importance_threshold = excluded.importance_threshold,
                dedup_enabled = excluded.dedup_enabled,
                dedup_threshold = excluded.dedup_threshold,
                archive_before_delete = excluded.archive_before_delete,
                archive_retention_days = excluded.archive_retention_days,
                auto_cleanup_enabled = excluded.auto_cleanup_enabled,
                updated_at = datetime('now')
            """,
            (
                project_path,
                config.get("retention_days", 90),
                config.get("min_relevance_score", 0.1),
                1 if config.get("keep_high_importance", True) else 0,
                config.get("importance_threshold", 7),
                1 if config.get("dedup_enabled", True) else 0,
                config.get("dedup_threshold", 0.95),
                1 if config.get("archive_before_delete", True) else 0,
                config.get("archive_retention_days", 365),
                1 if config.get("auto_cleanup_enabled", False) else 0
            )
        )
        self.db.conn.commit()
        return True

    async def run_cleanup(
        self,
        project_path: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Run full cleanup job.

        Args:
            project_path: Filter to specific project (None = all)
            dry_run: If True, only preview what would be cleaned

        Returns:
            Cleanup results with counts and details
        """
        config = await self.get_config(project_path)
        results = {
            "dry_run": dry_run,
            "project_path": project_path,
            "config": config,
            "low_relevance": {"count": 0, "ids": []},
            "expired": {"count": 0, "ids": []},
            "duplicates": {"count": 0, "groups": []},
            "total_archived": 0,
            "total_deleted": 0,
            "total_merged": 0
        }

        # 1. Clean up low-relevance memories
        low_rel_result = await self._cleanup_low_relevance(
            project_path=project_path,
            min_score=config["min_relevance_score"],
            keep_high_importance=config["keep_high_importance"],
            importance_threshold=config["importance_threshold"],
            archive=config["archive_before_delete"],
            dry_run=dry_run
        )
        results["low_relevance"] = low_rel_result
        results["total_archived"] += low_rel_result.get("archived", 0)
        results["total_deleted"] += low_rel_result.get("deleted", 0)

        # 2. Clean up expired memories
        expired_result = await self._cleanup_expired(
            project_path=project_path,
            retention_days=config["retention_days"],
            keep_high_importance=config["keep_high_importance"],
            importance_threshold=config["importance_threshold"],
            archive=config["archive_before_delete"],
            dry_run=dry_run
        )
        results["expired"] = expired_result
        results["total_archived"] += expired_result.get("archived", 0)
        results["total_deleted"] += expired_result.get("deleted", 0)

        # 3. Deduplicate memories
        if config["dedup_enabled"]:
            dedup_result = await self._deduplicate_memories(
                project_path=project_path,
                threshold=config["dedup_threshold"],
                archive=config["archive_before_delete"],
                dry_run=dry_run
            )
            results["duplicates"] = dedup_result
            results["total_merged"] += dedup_result.get("merged", 0)

        # 4. Log the cleanup
        if not dry_run:
            await self._log_cleanup(
                cleanup_type="full",
                project_path=project_path,
                archived=results["total_archived"],
                deleted=results["total_deleted"],
                merged=results["total_merged"],
                details=json.dumps(results)
            )

            # Update last cleanup timestamp
            cursor = self.db.conn.cursor()
            if project_path:
                cursor.execute(
                    """
                    UPDATE cleanup_config
                    SET last_cleanup_at = datetime('now')
                    WHERE project_path = ?
                    """,
                    (project_path,)
                )
            self.db.conn.commit()

        return results

    async def _cleanup_low_relevance(
        self,
        project_path: Optional[str],
        min_score: float,
        keep_high_importance: bool,
        importance_threshold: int,
        archive: bool,
        dry_run: bool
    ) -> Dict[str, Any]:
        """Clean up memories with low relevance scores."""
        cursor = self.db.conn.cursor()

        # Build query to find low-relevance memories
        query = """
            SELECT id, type, content, embedding, project_path, session_id,
                   importance, access_count, decay_factor, metadata,
                   created_at, last_accessed
            FROM memories
            WHERE 1=1
        """
        params = []

        if project_path:
            query += " AND project_path = ?"
            params.append(project_path)

        if keep_high_importance:
            query += " AND importance < ?"
            params.append(importance_threshold)

        cursor.execute(query, tuple(params))
        memories = [dict(row) for row in cursor.fetchall()]

        # Filter by calculated relevance score
        to_clean = []
        for mem in memories:
            score = self.db.calculate_relevance_score(
                importance=mem.get("importance", 5),
                created_at=mem.get("created_at"),
                last_accessed=mem.get("last_accessed"),
                access_count=mem.get("access_count", 0),
                decay_factor=mem.get("decay_factor", 1.0)
            )
            if score < min_score:
                mem["relevance_score"] = score
                to_clean.append(mem)

        result = {
            "count": len(to_clean),
            "ids": [m["id"] for m in to_clean],
            "archived": 0,
            "deleted": 0
        }

        if dry_run or not to_clean:
            return result

        # Archive and/or delete
        for mem in to_clean:
            if archive:
                await self._archive_memory(mem, reason="low_relevance")
                result["archived"] += 1

            cursor.execute("DELETE FROM memories WHERE id = ?", (mem["id"],))
            result["deleted"] += 1

        self.db.conn.commit()
        return result

    async def _cleanup_expired(
        self,
        project_path: Optional[str],
        retention_days: int,
        keep_high_importance: bool,
        importance_threshold: int,
        archive: bool,
        dry_run: bool
    ) -> Dict[str, Any]:
        """Clean up memories older than retention period."""
        cursor = self.db.conn.cursor()
        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()

        query = """
            SELECT id, type, content, embedding, project_path, session_id,
                   importance, access_count, decay_factor, metadata,
                   created_at, last_accessed
            FROM memories
            WHERE created_at < ?
        """
        params = [cutoff]

        if project_path:
            query += " AND project_path = ?"
            params.append(project_path)

        if keep_high_importance:
            query += " AND importance < ?"
            params.append(importance_threshold)

        cursor.execute(query, tuple(params))
        memories = [dict(row) for row in cursor.fetchall()]

        result = {
            "count": len(memories),
            "ids": [m["id"] for m in memories],
            "cutoff_date": cutoff,
            "archived": 0,
            "deleted": 0
        }

        if dry_run or not memories:
            return result

        for mem in memories:
            if archive:
                # Calculate relevance at archive time
                score = self.db.calculate_relevance_score(
                    importance=mem.get("importance", 5),
                    created_at=mem.get("created_at"),
                    last_accessed=mem.get("last_accessed"),
                    access_count=mem.get("access_count", 0),
                    decay_factor=mem.get("decay_factor", 1.0)
                )
                mem["relevance_score"] = score
                await self._archive_memory(mem, reason="expired")
                result["archived"] += 1

            cursor.execute("DELETE FROM memories WHERE id = ?", (mem["id"],))
            result["deleted"] += 1

        self.db.conn.commit()
        return result

    async def _deduplicate_memories(
        self,
        project_path: Optional[str],
        threshold: float,
        archive: bool,
        dry_run: bool
    ) -> Dict[str, Any]:
        """Find and merge duplicate memories."""
        cursor = self.db.conn.cursor()

        # Get memories with embeddings
        query = """
            SELECT id, type, content, embedding, project_path, session_id,
                   importance, access_count, created_at
            FROM memories
            WHERE embedding IS NOT NULL
        """
        params = []

        if project_path:
            query += " AND project_path = ?"
            params.append(project_path)

        query += " ORDER BY importance DESC, access_count DESC"

        cursor.execute(query, tuple(params))
        memories = [dict(row) for row in cursor.fetchall()]

        if len(memories) < 2:
            return {"count": 0, "groups": [], "merged": 0}

        # Find duplicate groups using greedy clustering
        groups = []
        used = set()

        for i, mem in enumerate(memories):
            if mem["id"] in used:
                continue

            emb1 = self._parse_embedding(mem.get("embedding"))
            if not emb1:
                continue

            group = [mem]
            used.add(mem["id"])

            for j, other in enumerate(memories[i+1:], start=i+1):
                if other["id"] in used:
                    continue

                emb2 = self._parse_embedding(other.get("embedding"))
                if not emb2:
                    continue

                similarity = self._cosine_similarity(emb1, emb2)
                if similarity >= threshold:
                    group.append(other)
                    used.add(other["id"])

            if len(group) > 1:
                groups.append(group)

        result = {
            "count": sum(len(g) - 1 for g in groups),  # Duplicates to remove
            "groups": [
                {
                    "keep_id": g[0]["id"],
                    "merge_ids": [m["id"] for m in g[1:]],
                    "content_preview": g[0]["content"][:100]
                }
                for g in groups
            ],
            "merged": 0
        }

        if dry_run or not groups:
            return result

        # Merge duplicates - keep highest importance, aggregate access count
        for group in groups:
            keep = group[0]
            duplicates = group[1:]

            # Aggregate stats
            total_access = keep.get("access_count", 0)
            for dup in duplicates:
                total_access += dup.get("access_count", 0)

                if archive:
                    await self._archive_memory(
                        dup,
                        reason="duplicate",
                        archived_by=f"merged_into_{keep['id']}"
                    )

                cursor.execute("DELETE FROM memories WHERE id = ?", (dup["id"],))
                result["merged"] += 1

            # Update the kept memory with aggregated access count
            cursor.execute(
                "UPDATE memories SET access_count = ? WHERE id = ?",
                (total_access, keep["id"])
            )

        self.db.conn.commit()
        return result

    async def _archive_memory(
        self,
        memory: Dict[str, Any],
        reason: str,
        archived_by: Optional[str] = None
    ):
        """Archive a memory before deletion."""
        cursor = self.db.conn.cursor()

        # Calculate expiration date
        config = await self.get_config(memory.get("project_path"))
        expires_at = (
            datetime.now() + timedelta(days=config["archive_retention_days"])
        ).isoformat()

        cursor.execute(
            """
            INSERT INTO memory_archive (
                original_id, type, content, embedding, project_path,
                session_id, importance, access_count, decay_factor,
                metadata, archive_reason, archived_by,
                relevance_score_at_archive, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                memory.get("id"),
                memory.get("type"),
                memory.get("content"),
                memory.get("embedding"),
                memory.get("project_path"),
                memory.get("session_id"),
                memory.get("importance"),
                memory.get("access_count"),
                memory.get("decay_factor"),
                memory.get("metadata"),
                reason,
                archived_by,
                memory.get("relevance_score"),
                expires_at
            )
        )

    async def _log_cleanup(
        self,
        cleanup_type: str,
        project_path: Optional[str],
        archived: int,
        deleted: int,
        merged: int,
        details: str
    ):
        """Log cleanup action for audit trail."""
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            INSERT INTO cleanup_log (
                cleanup_type, project_path, memories_archived,
                memories_deleted, memories_merged, details
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (cleanup_type, project_path, archived, deleted, merged, details)
        )
        self.db.conn.commit()

    async def get_archived_memories(
        self,
        project_path: Optional[str] = None,
        reason: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get archived memories for potential recovery."""
        cursor = self.db.conn.cursor()

        query = "SELECT * FROM memory_archive WHERE 1=1"
        params = []

        if project_path:
            query += " AND project_path = ?"
            params.append(project_path)

        if reason:
            query += " AND archive_reason = ?"
            params.append(reason)

        query += " ORDER BY archived_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, tuple(params))
        return [dict(row) for row in cursor.fetchall()]

    async def restore_memory(
        self,
        archive_id: int
    ) -> Dict[str, Any]:
        """Restore an archived memory."""
        cursor = self.db.conn.cursor()

        # Get archived memory
        cursor.execute("SELECT * FROM memory_archive WHERE id = ?", (archive_id,))
        archived = cursor.fetchone()

        if not archived:
            return {"success": False, "error": "Archived memory not found"}

        archived = dict(archived)

        # Restore to memories table
        cursor.execute(
            """
            INSERT INTO memories (
                type, content, embedding, project_path, session_id,
                importance, access_count, decay_factor, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                archived.get("type"),
                archived.get("content"),
                archived.get("embedding"),
                archived.get("project_path"),
                archived.get("session_id"),
                archived.get("importance"),
                archived.get("access_count"),
                archived.get("decay_factor"),
                archived.get("metadata")
            )
        )
        new_id = cursor.lastrowid

        # Remove from archive
        cursor.execute("DELETE FROM memory_archive WHERE id = ?", (archive_id,))
        self.db.conn.commit()

        return {
            "success": True,
            "restored_id": new_id,
            "original_id": archived.get("original_id"),
            "archive_reason": archived.get("archive_reason")
        }

    async def purge_expired_archives(self) -> Dict[str, Any]:
        """Permanently delete archives past their expiration date."""
        cursor = self.db.conn.cursor()

        # Count expired
        cursor.execute(
            "SELECT COUNT(*) as count FROM memory_archive WHERE expires_at < datetime('now')"
        )
        count = cursor.fetchone()["count"]

        if count > 0:
            cursor.execute(
                "DELETE FROM memory_archive WHERE expires_at < datetime('now')"
            )
            self.db.conn.commit()

        return {
            "success": True,
            "purged_count": count
        }

    async def get_cleanup_stats(self) -> Dict[str, Any]:
        """Get overall cleanup statistics."""
        cursor = self.db.conn.cursor()

        # Memory counts
        cursor.execute("SELECT COUNT(*) as count FROM memories")
        memory_count = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM memory_archive")
        archive_count = cursor.fetchone()["count"]

        # Recent cleanup log
        cursor.execute(
            """
            SELECT * FROM cleanup_log
            ORDER BY created_at DESC
            LIMIT 10
            """
        )
        recent_cleanups = [dict(row) for row in cursor.fetchall()]

        # Totals from logs
        cursor.execute(
            """
            SELECT
                SUM(memories_archived) as total_archived,
                SUM(memories_deleted) as total_deleted,
                SUM(memories_merged) as total_merged
            FROM cleanup_log
            """
        )
        totals = dict(cursor.fetchone())

        return {
            "current_memories": memory_count,
            "archived_memories": archive_count,
            "total_archived": totals.get("total_archived") or 0,
            "total_deleted": totals.get("total_deleted") or 0,
            "total_merged": totals.get("total_merged") or 0,
            "recent_cleanups": recent_cleanups
        }

    def _parse_embedding(self, embedding_str) -> Optional[List[float]]:
        """Parse embedding from string or list."""
        if not embedding_str:
            return None
        if isinstance(embedding_str, list):
            return embedding_str
        try:
            return json.loads(embedding_str)
        except:
            return None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        a = np.array(vec1)
        b = np.array(vec2)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))


# Global instance
_cleanup: Optional[CleanupService] = None


def get_cleanup_service(db, embeddings=None) -> CleanupService:
    """Get the global cleanup service instance."""
    global _cleanup
    if _cleanup is None:
        _cleanup = CleanupService(db, embeddings)
    return _cleanup
