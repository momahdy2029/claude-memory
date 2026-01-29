"""Confidence scoring and verification service for memories.

Calculates confidence scores based on:
- Age (newer = higher confidence)
- Access count (frequently accessed = higher)
- Verification status
- Source reliability
- Contradiction checks
"""
import time
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta


class ConfidenceService:
    """Service for memory confidence scoring and verification.

    Confidence is a value 0-1 representing how reliable a memory is.
    """

    def __init__(self, db, embeddings):
        self.db = db
        self.embeddings = embeddings

        # Weights for confidence calculation
        self.weights = {
            "age": 0.20,           # How recent
            "access": 0.15,        # How often accessed
            "importance": 0.15,    # User-assigned importance
            "verification": 0.25,  # Verified status
            "consistency": 0.25,   # No contradictions
        }

        # Age decay parameters
        self.age_half_life_days = 90  # 50% confidence after 90 days

    def _calculate_age_score(self, created_at: str) -> float:
        """Calculate age-based confidence (exponential decay)."""
        try:
            created = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
            age_days = (datetime.now(created.tzinfo or None) - created).days
            if age_days < 0:
                age_days = 0

            # Exponential decay
            half_life = self.age_half_life_days
            score = 0.5 ** (age_days / half_life)
            return max(0.1, min(1.0, score))  # Clamp to [0.1, 1.0]
        except:
            return 0.5  # Default for unparseable dates

    def _calculate_access_score(self, access_count: int) -> float:
        """Calculate access-based confidence (logarithmic growth)."""
        if access_count <= 0:
            return 0.3  # Baseline for never accessed
        # Logarithmic scale - more accesses = higher confidence
        import math
        score = 0.3 + 0.7 * (1 - 1 / (1 + math.log(access_count + 1)))
        return min(1.0, score)

    def _calculate_importance_score(self, importance: int) -> float:
        """Normalize importance (1-10) to confidence contribution."""
        return importance / 10.0

    async def calculate_confidence(
        self,
        memory_id: int
    ) -> Dict[str, Any]:
        """Calculate confidence score for a memory.

        Returns detailed breakdown of confidence components.
        """
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT id, content, type, importance, created_at,
                   access_count, decay_factor, metadata
            FROM memories WHERE id = ?
        """, [memory_id])

        row = cursor.fetchone()
        if not row:
            return {"success": False, "error": "Memory not found"}

        memory = {
            "id": row[0],
            "content": row[1],
            "type": row[2],
            "importance": row[3] or 5,
            "created_at": row[4],
            "access_count": row[5] or 0,
            "decay_factor": row[6] or 1.0,
            "metadata": row[7]
        }

        # Calculate component scores
        age_score = self._calculate_age_score(memory["created_at"])
        access_score = self._calculate_access_score(memory["access_count"])
        importance_score = self._calculate_importance_score(memory["importance"])

        # Check verification status from metadata
        verified = False
        if memory.get("metadata"):
            try:
                import json
                meta = json.loads(memory["metadata"])
                verified = meta.get("verified", False)
            except:
                pass
        verification_score = 1.0 if verified else 0.5

        # Check for contradictions (anchors)
        consistency_score = await self._check_consistency(memory_id, memory["content"])

        # Apply decay factor
        decay = memory["decay_factor"]

        # Weighted sum
        raw_confidence = (
            self.weights["age"] * age_score +
            self.weights["access"] * access_score +
            self.weights["importance"] * importance_score +
            self.weights["verification"] * verification_score +
            self.weights["consistency"] * consistency_score
        ) * decay

        confidence = min(1.0, max(0.0, raw_confidence))

        return {
            "success": True,
            "memory_id": memory_id,
            "confidence": round(confidence, 3),
            "breakdown": {
                "age": round(age_score, 3),
                "access": round(access_score, 3),
                "importance": round(importance_score, 3),
                "verification": round(verification_score, 3),
                "consistency": round(consistency_score, 3),
                "decay_factor": round(decay, 3)
            },
            "verified": verified,
            "interpretation": self._interpret_confidence(confidence)
        }

    async def _check_consistency(self, memory_id: int, content: str) -> float:
        """Check if memory is consistent with anchors."""
        # Look for any conflicts involving this memory
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM anchor_conflicts
            WHERE (anchor1_id = ? OR anchor2_id = ?)
            AND status = 'unresolved'
        """, [memory_id, memory_id])

        conflicts = cursor.fetchone()[0]

        if conflicts > 0:
            return 0.3  # Low consistency if conflicts exist
        return 1.0  # Full consistency if no conflicts

    def _interpret_confidence(self, confidence: float) -> str:
        """Human-readable interpretation of confidence score."""
        if confidence >= 0.9:
            return "Very high - this memory is reliable"
        elif confidence >= 0.7:
            return "High - likely accurate"
        elif confidence >= 0.5:
            return "Moderate - use with caution"
        elif confidence >= 0.3:
            return "Low - may be outdated or unverified"
        else:
            return "Very low - consider verification"

    async def verify_memory(
        self,
        memory_id: int,
        verified: bool = True,
        verified_by: str = "user"
    ) -> Dict[str, Any]:
        """Mark a memory as verified or unverified."""
        import json

        cursor = self.db.conn.cursor()

        # Get current metadata
        cursor.execute("SELECT metadata FROM memories WHERE id = ?", [memory_id])
        row = cursor.fetchone()
        if not row:
            return {"success": False, "error": "Memory not found"}

        try:
            metadata = json.loads(row[0]) if row[0] else {}
        except:
            metadata = {}

        metadata["verified"] = verified
        metadata["verified_by"] = verified_by
        metadata["verified_at"] = datetime.now().isoformat()

        cursor.execute(
            "UPDATE memories SET metadata = ? WHERE id = ?",
            [json.dumps(metadata), memory_id]
        )
        self.db.conn.commit()

        # Recalculate confidence
        new_confidence = await self.calculate_confidence(memory_id)

        return {
            "success": True,
            "memory_id": memory_id,
            "verified": verified,
            "new_confidence": new_confidence.get("confidence")
        }

    async def mark_outdated(
        self,
        memory_id: int,
        reason: str = "manually marked"
    ) -> Dict[str, Any]:
        """Mark a memory as outdated (reduces confidence significantly)."""
        cursor = self.db.conn.cursor()

        # Set decay factor to low value
        cursor.execute(
            "UPDATE memories SET decay_factor = 0.3 WHERE id = ?",
            [memory_id]
        )
        self.db.conn.commit()

        return {
            "success": True,
            "memory_id": memory_id,
            "marked_outdated": True,
            "reason": reason
        }

    async def get_low_confidence_memories(
        self,
        project_path: Optional[str] = None,
        threshold: float = 0.5,
        limit: int = 20
    ) -> Dict[str, Any]:
        """Get memories with low confidence that may need verification."""
        cursor = self.db.conn.cursor()

        query = """
            SELECT id, content, type, importance, created_at, access_count, decay_factor
            FROM memories
            WHERE decay_factor < 0.8 OR access_count = 0
        """
        params = []

        if project_path:
            query += " AND project_path = ?"
            params.append(project_path)

        query += " ORDER BY decay_factor ASC, access_count ASC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for row in rows:
            memory = {
                "id": row[0],
                "content": row[1][:200],
                "type": row[2],
                "importance": row[3],
                "created_at": row[4],
                "access_count": row[5],
                "decay_factor": row[6]
            }

            # Calculate full confidence
            conf = await self.calculate_confidence(row[0])
            if conf.get("confidence", 1.0) <= threshold:
                memory["confidence"] = conf.get("confidence")
                memory["interpretation"] = conf.get("interpretation")
                results.append(memory)

        return {
            "success": True,
            "low_confidence_memories": results,
            "count": len(results),
            "threshold": threshold
        }


# Global instance
_confidence_service: Optional[ConfidenceService] = None


def get_confidence_service(db, embeddings) -> ConfidenceService:
    """Get the global confidence service."""
    global _confidence_service
    if _confidence_service is None:
        _confidence_service = ConfidenceService(db, embeddings)
    return _confidence_service
