"""Hierarchical Memory Tier Manager - CLaRa-inspired multi-stage processing.

Manages three tiers of memories with different search strategies:
- Hot: Recent, high-importance, frequently accessed. Full semantic + keyword search, flat FAISS.
- Warm: Older but relevant. Semantic search on compressed content, IVF FAISS.
- Cold: Archived, low importance. Keyword-only search, no FAISS index.

Permanent types (decision, preference, code) with importance >= 5 stay hot.
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from config import config
from services.memory_decay import DECAY_LIFESPANS

logger = logging.getLogger(__name__)

# Tier constants
TIER_HOT = 'hot'
TIER_WARM = 'warm'
TIER_COLD = 'cold'

# Permanent types that resist demotion
PERMANENT_TYPES = {'decision', 'preference', 'code'}


class TierManager:
    """Manages memory tier promotion and demotion.

    Scoring algorithm considers:
    - Age (days since creation)
    - Importance (1-10)
    - Access frequency (access_count)
    - Memory type (permanent vs ephemeral)
    - Decay factor
    """

    def __init__(self, db):
        self.db = db
        self.hot_max_age = config.TIER_HOT_MAX_AGE_DAYS
        self.hot_min_importance = config.TIER_HOT_MIN_IMPORTANCE
        self.warm_max_age = config.TIER_WARM_MAX_AGE_DAYS

    def evaluate_tier(self, memory: dict) -> str:
        """Evaluate which tier a memory belongs in.

        Args:
            memory: Dict with type, importance, created_at, access_count, decay_factor

        Returns:
            Tier string: 'hot', 'warm', or 'cold'
        """
        memory_type = memory.get('type', 'chunk')
        importance = memory.get('importance', 5)
        access_count = memory.get('access_count', 0) or 0
        decay_factor = memory.get('decay_factor', 1.0) or 1.0

        # Permanent types with importance >= 5 always stay hot
        if memory_type in PERMANENT_TYPES and importance >= 5:
            return TIER_HOT

        # Calculate age
        created_at = memory.get('created_at')
        age_days = self._calculate_age_days(created_at)

        # Hot tier criteria:
        # - Recent (< hot_max_age days) OR
        # - High importance (>= hot_min_importance) OR
        # - Frequently accessed (access_count >= 5 in last 14 days)
        if age_days < self.hot_max_age:
            return TIER_HOT
        if importance >= self.hot_min_importance:
            return TIER_HOT
        if access_count >= 5 and age_days < self.hot_max_age * 2:
            return TIER_HOT

        # Warm tier criteria:
        # - Age < warm_max_age AND (importance >= 3 OR access_count >= 2)
        if age_days < self.warm_max_age:
            if importance >= 3 or access_count >= 2:
                return TIER_WARM
            # Low importance, low access but still within warm window
            if decay_factor > 0.3:
                return TIER_WARM

        # Cold: everything else
        return TIER_COLD

    def _calculate_age_days(self, created_at: Optional[str]) -> float:
        """Calculate age in days from a timestamp string."""
        if not created_at:
            return 0.0
        try:
            created_dt = datetime.fromisoformat(
                created_at.replace('Z', '+00:00')
            ).replace(tzinfo=None)
            return (datetime.now() - created_dt).total_seconds() / 86400.0
        except (ValueError, TypeError, AttributeError):
            return 0.0

    async def run_tier_maintenance(self, skip_recent_hours: int = 24) -> Dict[str, Any]:
        """Evaluate and update tiers for all memories.

        Combines with decay evaluation for efficiency (single pass over all memories).

        Args:
            skip_recent_hours: Skip memories whose tier was evaluated within this window

        Returns:
            Dict with maintenance statistics
        """
        cursor = self.db.conn.cursor()

        # Fetch all memories that need tier evaluation
        # Skip those evaluated recently (tier_changed_at within skip window)
        cutoff = (datetime.now() - timedelta(hours=skip_recent_hours)).isoformat()

        cursor.execute("""
            SELECT id, type, importance, access_count, decay_factor, created_at,
                   last_accessed, tier, tier_changed_at
            FROM memories
            WHERE tier_changed_at IS NULL OR tier_changed_at < ?
        """, (cutoff,))

        rows = cursor.fetchall()

        stats = {
            'evaluated': 0,
            'promoted': 0,
            'demoted': 0,
            'unchanged': 0,
            'tier_counts': {TIER_HOT: 0, TIER_WARM: 0, TIER_COLD: 0}
        }

        updates = []

        for row in rows:
            stats['evaluated'] += 1
            memory_dict = dict(row)
            new_tier = self.evaluate_tier(memory_dict)
            old_tier = row['tier'] or TIER_HOT  # Default to hot for unmigrated

            stats['tier_counts'][new_tier] = stats['tier_counts'].get(new_tier, 0) + 1

            if new_tier != old_tier:
                updates.append((new_tier, datetime.now().isoformat(), row['id']))
                if self._tier_rank(new_tier) < self._tier_rank(old_tier):
                    stats['promoted'] += 1
                else:
                    stats['demoted'] += 1
            else:
                stats['unchanged'] += 1
                # Still update tier_changed_at to avoid re-evaluation
                updates.append((new_tier, datetime.now().isoformat(), row['id']))

        # Batch update
        if updates:
            cursor.executemany(
                "UPDATE memories SET tier = ?, tier_changed_at = ? WHERE id = ?",
                updates
            )
            self.db.conn.commit()

        stats['timestamp'] = datetime.now().isoformat()
        return stats

    def _tier_rank(self, tier: str) -> int:
        """Numeric rank for tier comparison (lower = hotter)."""
        return {TIER_HOT: 0, TIER_WARM: 1, TIER_COLD: 2}.get(tier, 1)

    async def promote_on_access(self, memory_id: int) -> Optional[str]:
        """Re-evaluate and promote a memory's tier when it is accessed.

        Called after search results are returned to ensure frequently-accessed
        memories don't stay cold/warm until the next batch maintenance run.

        Args:
            memory_id: ID of the accessed memory

        Returns:
            New tier if promoted, None if unchanged
        """
        cursor = self.db.conn.cursor()
        cursor.execute(
            "SELECT id, type, importance, access_count, decay_factor, created_at, tier FROM memories WHERE id = ?",
            (memory_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None

        current_tier = row['tier'] or TIER_HOT
        new_tier = self.evaluate_tier(dict(row))

        if self._tier_rank(new_tier) < self._tier_rank(current_tier):
            cursor.execute(
                "UPDATE memories SET tier = ?, tier_changed_at = ? WHERE id = ?",
                (new_tier, datetime.now().isoformat(), memory_id)
            )
            self.db.conn.commit()
            logger.info(f"Memory {memory_id} promoted from {current_tier} to {new_tier}")
            return new_tier

        return None

    async def get_tier_stats(self) -> Dict[str, Any]:
        """Get distribution of memories across tiers."""
        cursor = self.db.conn.cursor()

        cursor.execute("""
            SELECT COALESCE(tier, 'hot') as tier, COUNT(*) as count,
                   AVG(importance) as avg_importance,
                   AVG(access_count) as avg_access_count
            FROM memories
            GROUP BY COALESCE(tier, 'hot')
        """)

        tiers = {}
        total = 0
        for row in cursor.fetchall():
            tier = row['tier']
            count = row['count']
            total += count
            tiers[tier] = {
                'count': count,
                'avg_importance': round(row['avg_importance'] or 0, 2),
                'avg_access_count': round(row['avg_access_count'] or 0, 2)
            }

        return {
            'total_memories': total,
            'tiers': tiers,
            'timestamp': datetime.now().isoformat()
        }
