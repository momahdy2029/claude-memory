"""Memory Decay Service - Type-based lifespan management for memories.

Implements automatic decay of memory relevance based on type-specific lifespans,
access patterns, and age. Permanent types (decision, preference, code) never decay.
Temporary types (session, chunk, error) decay over configurable lifespans.

Decay formula:
    relevance_score = base_score * decay_multiplier * access_boost
    decay_multiplier = max(0, 1 - (age_days / lifespan_days))  [non-permanent only]
    access_boost = 1 + (0.1 * min(access_count, 10))  [caps at 2x]
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# Type-based lifespans in days. None means permanent (never decays).
DECAY_LIFESPANS = {
    "decision": None,      # permanent - architectural choices persist
    "preference": None,    # permanent - user preferences persist
    "code": None,          # permanent - reusable patterns persist
    "error": 90,           # 3 months - errors become stale
    "session": 7,          # 1 week - session context is ephemeral
    "chunk": 30,           # 1 month - general chunks decay moderately
}

# Default lifespan for unknown types
DEFAULT_LIFESPAN_DAYS = 30


class MemoryDecayService:
    """Manages memory decay based on type-specific lifespans and access patterns.

    Permanent types (decision, preference, code) are never decayed.
    Temporary types have configurable lifespans after which they are archived.
    Frequently accessed memories resist decay through an access boost.
    """

    def __init__(self, db, archive_threshold: float = 0.1):
        """Initialize the decay service.

        Args:
            db: DatabaseService instance (synchronous sqlite3 connection via db.conn)
            archive_threshold: Minimum decay score before archiving (default 0.1)
        """
        self.db = db
        self.archive_threshold = archive_threshold

    def calculate_decay_score(self, memory: dict) -> float:
        """Calculate the current relevance score for a memory based on decay.

        Args:
            memory: Dict with keys: type, created_at, access_count, importance, confidence

        Returns:
            Float relevance score. 1.0 = fully relevant, 0.0 = fully decayed.
            Permanent types always return 1.0.
        """
        memory_type = memory.get("type", "chunk")
        lifespan = DECAY_LIFESPANS.get(memory_type, DEFAULT_LIFESPAN_DAYS)

        # Permanent types never decay
        if lifespan is None:
            return 1.0

        # Calculate age in days
        created_at = memory.get("created_at")
        if not created_at:
            return 1.0

        try:
            created_dt = datetime.fromisoformat(
                created_at.replace('Z', '+00:00')
            ).replace(tzinfo=None)
            age_days = (datetime.now() - created_dt).total_seconds() / 86400.0
        except (ValueError, TypeError, AttributeError):
            return 1.0

        # Decay multiplier: linear decay from 1.0 to 0.0 over the lifespan
        decay_multiplier = max(0.0, 1.0 - (age_days / lifespan))

        # Access boost: frequently accessed memories resist decay
        # Caps at 2x boost (access_count=10 gives 1 + 0.1*10 = 2.0)
        access_count = memory.get("access_count", 0) or 0
        access_boost = 1.0 + (0.1 * min(access_count, 10))

        # Final score: decay_multiplier * access_boost, capped at 1.0
        score = min(decay_multiplier * access_boost, 1.0)

        return round(score, 4)

    async def apply_decay(self, update_tiers: bool = True) -> Dict[str, Any]:
        """Run decay across all non-permanent memories and archive those below threshold.

        Optionally combines with tier evaluation in a single pass for efficiency.

        Args:
            update_tiers: If True, also evaluate and update memory tiers

        Returns:
            Dict with statistics about the decay run.
        """
        cursor = self.db.conn.cursor()

        # Only process types with finite lifespans
        decayable_types = [
            mtype for mtype, lifespan in DECAY_LIFESPANS.items()
            if lifespan is not None
        ]

        if not decayable_types:
            return {
                "success": True,
                "memories_evaluated": 0,
                "memories_archived": 0,
                "message": "No decayable types configured"
            }

        # Fetch all non-permanent memories
        placeholders = ",".join("?" * len(decayable_types))
        cursor.execute(f"""
            SELECT id, type, content, embedding, project_path, session_id,
                   importance, access_count, decay_factor, metadata,
                   confidence, created_at, last_accessed, tier, tier_changed_at
            FROM memories
            WHERE type IN ({placeholders})
        """, decayable_types)

        rows = cursor.fetchall()

        evaluated = 0
        archived = 0
        updated = 0
        tier_changes = 0
        scores_by_type = {}

        # Lazy-load tier manager if tier updates requested
        tier_manager = None
        if update_tiers:
            try:
                from services.tier_manager import TierManager
                tier_manager = TierManager(self.db)
            except ImportError:
                logger.warning("TierManager not available, skipping tier updates")

        for row in rows:
            evaluated += 1
            memory_dict = dict(row)
            decay_score = self.calculate_decay_score(memory_dict)
            memory_type = row["type"]

            # Track scores by type for stats
            if memory_type not in scores_by_type:
                scores_by_type[memory_type] = {
                    "count": 0,
                    "total_score": 0.0,
                    "archived": 0,
                    "min_score": 1.0,
                    "max_score": 0.0
                }
            type_stats = scores_by_type[memory_type]
            type_stats["count"] += 1
            type_stats["total_score"] += decay_score
            type_stats["min_score"] = min(type_stats["min_score"], decay_score)
            type_stats["max_score"] = max(type_stats["max_score"], decay_score)

            if decay_score < self.archive_threshold:
                # Archive the memory
                try:
                    cursor.execute("""
                        INSERT INTO memory_archive
                        (original_id, type, content, embedding, project_path, session_id,
                         importance, access_count, decay_factor, metadata,
                         archive_reason, relevance_score_at_archive)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'decay_expired', ?)
                    """, (
                        row["id"], row["type"], row["content"], row["embedding"],
                        row["project_path"], row["session_id"],
                        row["importance"], row["access_count"],
                        row["decay_factor"], row["metadata"],
                        decay_score
                    ))

                    # Delete from active memories
                    cursor.execute("DELETE FROM memories WHERE id = ?", (row["id"],))
                    archived += 1
                    type_stats["archived"] += 1
                except Exception as e:
                    logger.error(f"Failed to archive memory {row['id']}: {e}")
            else:
                # Combined update: decay_factor + tier in single UPDATE
                new_tier = None
                if tier_manager:
                    new_tier = tier_manager.evaluate_tier(memory_dict)
                    old_tier = row.get('tier') or 'hot'
                    if new_tier != old_tier:
                        tier_changes += 1

                if new_tier:
                    cursor.execute("""
                        UPDATE memories
                        SET decay_factor = ?, tier = ?, tier_changed_at = ?
                        WHERE id = ?
                    """, (decay_score, new_tier, datetime.now().isoformat(), row["id"]))
                else:
                    cursor.execute("""
                        UPDATE memories
                        SET decay_factor = ?
                        WHERE id = ?
                    """, (decay_score, row["id"]))
                updated += 1

        self.db.conn.commit()

        # Build average scores
        for type_stats in scores_by_type.values():
            if type_stats["count"] > 0:
                type_stats["avg_score"] = round(
                    type_stats["total_score"] / type_stats["count"], 4
                )
            del type_stats["total_score"]

        result = {
            "success": True,
            "memories_evaluated": evaluated,
            "memories_archived": archived,
            "memories_updated": updated,
            "archive_threshold": self.archive_threshold,
            "scores_by_type": scores_by_type,
            "timestamp": datetime.now().isoformat()
        }

        if update_tiers:
            result["tier_changes"] = tier_changes

        return result

    async def boost_on_access(self, memory_id: int) -> Dict[str, Any]:
        """Called when a memory is accessed. Increments access_count and updates last_accessed.

        Args:
            memory_id: ID of the memory being accessed

        Returns:
            Dict with updated access stats
        """
        cursor = self.db.conn.cursor()

        cursor.execute("""
            UPDATE memories
            SET access_count = COALESCE(access_count, 0) + 1,
                last_accessed = datetime('now')
            WHERE id = ?
        """, (memory_id,))

        self.db.conn.commit()

        if cursor.rowcount == 0:
            return {"success": False, "error": f"Memory {memory_id} not found"}

        # Fetch updated stats
        cursor.execute("""
            SELECT id, type, access_count, last_accessed, created_at, importance, confidence
            FROM memories WHERE id = ?
        """, (memory_id,))
        row = cursor.fetchone()

        if not row:
            return {"success": False, "error": f"Memory {memory_id} not found after update"}

        decay_score = self.calculate_decay_score(dict(row))

        return {
            "success": True,
            "memory_id": memory_id,
            "access_count": row["access_count"],
            "last_accessed": row["last_accessed"],
            "current_decay_score": decay_score
        }

    async def get_decay_stats(self) -> Dict[str, Any]:
        """Return statistics on decayed, active, and permanent memories.

        Returns:
            Dict with comprehensive decay statistics
        """
        cursor = self.db.conn.cursor()

        # Count by type
        cursor.execute("""
            SELECT type, COUNT(*) as count
            FROM memories
            GROUP BY type
        """)
        type_counts = {row["type"]: row["count"] for row in cursor.fetchall()}

        # Separate permanent vs decayable
        permanent_types = [
            mtype for mtype, lifespan in DECAY_LIFESPANS.items()
            if lifespan is None
        ]
        decayable_types = [
            mtype for mtype, lifespan in DECAY_LIFESPANS.items()
            if lifespan is not None
        ]

        permanent_count = sum(
            type_counts.get(t, 0) for t in permanent_types
        )
        decayable_count = sum(
            type_counts.get(t, 0) for t in decayable_types
        )
        # Unknown types also count as decayable
        known_types = set(DECAY_LIFESPANS.keys())
        unknown_count = sum(
            count for t, count in type_counts.items() if t not in known_types
        )

        # Get archived count (from decay)
        cursor.execute("""
            SELECT COUNT(*) as count
            FROM memory_archive
            WHERE archive_reason = 'decay_expired'
        """)
        archived_by_decay = cursor.fetchone()["count"]

        # Calculate decay scores for all decayable memories
        at_risk = 0  # Score between threshold and 0.3
        healthy = 0  # Score above 0.3

        if decayable_types:
            placeholders = ",".join("?" * len(decayable_types))
            cursor.execute(f"""
                SELECT type, access_count, created_at, importance, confidence
                FROM memories
                WHERE type IN ({placeholders})
            """, decayable_types)

            for row in cursor.fetchall():
                score = self.calculate_decay_score(dict(row))
                if score < 0.3:
                    at_risk += 1
                else:
                    healthy += 1

        # Lifespan info for reference
        lifespan_info = {}
        for mtype, lifespan in DECAY_LIFESPANS.items():
            lifespan_info[mtype] = {
                "lifespan_days": lifespan if lifespan is not None else "permanent",
                "active_count": type_counts.get(mtype, 0)
            }

        return {
            "total_memories": sum(type_counts.values()),
            "permanent_count": permanent_count,
            "decayable_count": decayable_count + unknown_count,
            "archived_by_decay": archived_by_decay,
            "at_risk_count": at_risk,
            "healthy_count": healthy,
            "archive_threshold": self.archive_threshold,
            "type_details": lifespan_info,
            "type_counts": type_counts,
            "timestamp": datetime.now().isoformat()
        }


def calculate_search_decay_multiplier(memory: dict) -> float:
    """Calculate a decay multiplier suitable for search ranking.

    This is a standalone function that can be called from search_similar()
    without needing the full MemoryDecayService instance.

    Args:
        memory: Dict with at least 'type', 'created_at', 'access_count'

    Returns:
        Float multiplier between 0.0 and 1.0. Permanent types return 1.0.
    """
    memory_type = memory.get("type", "chunk")
    lifespan = DECAY_LIFESPANS.get(memory_type, DEFAULT_LIFESPAN_DAYS)

    # Permanent types: no decay penalty
    if lifespan is None:
        return 1.0

    # Calculate age
    created_at = memory.get("created_at")
    if not created_at:
        return 1.0

    try:
        created_dt = datetime.fromisoformat(
            created_at.replace('Z', '+00:00')
        ).replace(tzinfo=None)
        age_days = (datetime.now() - created_dt).total_seconds() / 86400.0
    except (ValueError, TypeError, AttributeError):
        return 1.0

    # Linear decay
    decay_multiplier = max(0.0, 1.0 - (age_days / lifespan))

    # Access boost
    access_count = memory.get("access_count", 0) or 0
    access_boost = 1.0 + (0.1 * min(access_count, 10))

    return min(decay_multiplier * access_boost, 1.0)
