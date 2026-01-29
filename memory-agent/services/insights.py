"""Cross-session learning and insight aggregation service.

Analyzes memories across sessions to identify patterns, recurring issues,
and aggregated learnings that can improve future interactions.
"""
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict


class InsightsService:
    """Service for generating and managing cross-session insights.

    Features:
    - Error pattern detection (similar errors across sessions)
    - Decision aggregation (same problem -> same solution patterns)
    - User correction detection (Claude blind spots)
    - High-value memory identification
    - CLAUDE.md improvement suggestions
    """

    def __init__(self, db, embeddings):
        self.db = db
        self.embeddings = embeddings

    async def aggregate_error_patterns(
        self,
        days_back: int = 30,
        min_occurrences: int = 2,
        similarity_threshold: float = 0.85
    ) -> List[Dict[str, Any]]:
        """Find recurring error patterns across sessions.

        Groups similar errors by embedding similarity and extracts
        common resolution patterns.

        Returns:
            List of error pattern insights
        """
        # Get recent error memories
        cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()
        errors = await self.db.execute_query(
            """
            SELECT id, content, embedding, session_id, outcome, project_path,
                   tech_stack, created_at
            FROM memories
            WHERE type = 'error'
            AND created_at > ?
            AND embedding IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 500
            """,
            (cutoff,)
        )

        if not errors or len(errors) < min_occurrences:
            return []

        # Group similar errors
        clusters = await self._cluster_by_embedding(
            errors, similarity_threshold
        )

        insights = []
        for cluster in clusters:
            if len(cluster) < min_occurrences:
                continue

            # Extract common elements
            content_samples = [e["content"][:200] for e in cluster[:3]]
            sessions = list(set(e["session_id"] for e in cluster if e["session_id"]))
            projects = list(set(e["project_path"] for e in cluster if e["project_path"]))

            # Find successful resolutions
            resolved = [e for e in cluster if e.get("outcome") and "fix" in e.get("outcome", "").lower()]

            insight = {
                "insight_type": "recurring_error",
                "title": f"Recurring error pattern ({len(cluster)} occurrences)",
                "description": self._summarize_cluster(cluster, "error"),
                "evidence_count": len(cluster),
                "evidence_ids": json.dumps([e["id"] for e in cluster]),
                "source_sessions": json.dumps(sessions[:10]),
                "confidence": min(0.9, 0.5 + (len(cluster) * 0.1)),
                "impact_score": min(10, 5 + len(cluster)),
                "project_path": projects[0] if len(projects) == 1 else None,
                "category": "error_pattern",
                "resolution_found": len(resolved) > 0,
                "sample_content": content_samples
            }
            insights.append(insight)

        return insights

    async def aggregate_decision_patterns(
        self,
        days_back: int = 60,
        min_occurrences: int = 2,
        similarity_threshold: float = 0.80
    ) -> List[Dict[str, Any]]:
        """Find recurring decision patterns (same problem -> same solution).

        Identifies when Claude makes the same type of decision repeatedly,
        which could be codified into CLAUDE.md rules.

        Returns:
            List of decision pattern insights
        """
        cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()
        decisions = await self.db.execute_query(
            """
            SELECT id, content, embedding, session_id, outcome, success,
                   project_path, tech_stack, created_at
            FROM memories
            WHERE type = 'decision'
            AND created_at > ?
            AND embedding IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 500
            """,
            (cutoff,)
        )

        if not decisions or len(decisions) < min_occurrences:
            return []

        # Group similar decisions
        clusters = await self._cluster_by_embedding(
            decisions, similarity_threshold
        )

        insights = []
        for cluster in clusters:
            if len(cluster) < min_occurrences:
                continue

            # Calculate success rate for this decision type
            successful = sum(1 for d in cluster if d.get("success") == 1)
            success_rate = successful / len(cluster) if cluster else 0

            # Extract tech context
            tech_stacks = []
            for d in cluster:
                if d.get("tech_stack"):
                    try:
                        stacks = json.loads(d["tech_stack"]) if isinstance(d["tech_stack"], str) else d["tech_stack"]
                        tech_stacks.extend(stacks if isinstance(stacks, list) else [stacks])
                    except:
                        pass

            insight = {
                "insight_type": "decision_pattern",
                "title": f"Recurring decision pattern ({len(cluster)} times)",
                "description": self._summarize_cluster(cluster, "decision"),
                "evidence_count": len(cluster),
                "evidence_ids": json.dumps([d["id"] for d in cluster]),
                "source_sessions": json.dumps(list(set(d["session_id"] for d in cluster if d["session_id"]))[:10]),
                "confidence": min(0.95, 0.5 + (success_rate * 0.3) + (len(cluster) * 0.05)),
                "impact_score": min(10, 5 + (success_rate * 3)),
                "category": "decision_pattern",
                "success_rate": success_rate,
                "tech_context": json.dumps(list(set(tech_stacks))[:5]) if tech_stacks else None,
                "sample_content": [d["content"][:200] for d in cluster[:3]]
            }
            insights.append(insight)

        return insights

    async def detect_correction_patterns(
        self,
        days_back: int = 30
    ) -> List[Dict[str, Any]]:
        """Detect patterns where user had to correct Claude.

        These indicate blind spots that should be addressed in CLAUDE.md.

        Returns:
            List of correction pattern insights
        """
        cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()

        # Look for memories with negative user feedback or failed outcomes
        corrections = await self.db.execute_query(
            """
            SELECT id, content, type, session_id, outcome, user_feedback,
                   project_path, agent_type, created_at
            FROM memories
            WHERE created_at > ?
            AND (
                user_feedback LIKE '%wrong%' OR
                user_feedback LIKE '%incorrect%' OR
                user_feedback LIKE '%no%' OR
                user_feedback LIKE '%fix%' OR
                outcome LIKE '%failed%' OR
                outcome LIKE '%error%' OR
                success = 0
            )
            ORDER BY created_at DESC
            LIMIT 200
            """,
            (cutoff,)
        )

        if not corrections:
            return []

        # Group by type/pattern
        by_type = defaultdict(list)
        for c in corrections:
            key = c.get("type", "unknown")
            by_type[key].append(c)

        insights = []
        for memory_type, items in by_type.items():
            if len(items) < 2:
                continue

            sessions = list(set(i["session_id"] for i in items if i["session_id"]))

            insight = {
                "insight_type": "correction_pattern",
                "title": f"Repeated corrections in {memory_type} ({len(items)} times)",
                "description": f"User frequently corrected Claude on {memory_type} tasks. "
                              f"Consider adding specific guidance to CLAUDE.md.",
                "evidence_count": len(items),
                "evidence_ids": json.dumps([i["id"] for i in items]),
                "source_sessions": json.dumps(sessions[:10]),
                "confidence": min(0.8, 0.4 + (len(items) * 0.1)),
                "impact_score": min(10, 6 + len(items)),
                "category": "blind_spot",
                "memory_type": memory_type,
                "sample_feedback": [i.get("user_feedback", i.get("outcome", ""))[:100]
                                   for i in items[:3] if i.get("user_feedback") or i.get("outcome")]
            }
            insights.append(insight)

        return insights

    async def identify_high_value_memories(
        self,
        days_back: int = 90,
        min_access_count: int = 3
    ) -> List[Dict[str, Any]]:
        """Identify frequently accessed memories (high-value content).

        Returns:
            List of high-value memory insights
        """
        cutoff = (datetime.now() - timedelta(days=days_back)).isoformat()

        high_value = await self.db.execute_query(
            """
            SELECT id, content, type, access_count, importance, project_path,
                   tech_stack, session_id, created_at
            FROM memories
            WHERE created_at > ?
            AND access_count >= ?
            ORDER BY access_count DESC, importance DESC
            LIMIT 50
            """,
            (cutoff, min_access_count)
        )

        if not high_value:
            return []

        insights = []
        for mem in high_value:
            insight = {
                "insight_type": "high_value_memory",
                "title": f"High-value {mem['type']} (accessed {mem['access_count']} times)",
                "description": mem["content"][:300],
                "evidence_count": 1,
                "evidence_ids": json.dumps([mem["id"]]),
                "confidence": 0.9,
                "impact_score": min(10, mem["importance"] + (mem["access_count"] * 0.5)),
                "category": "valuable_content",
                "access_count": mem["access_count"],
                "memory_type": mem["type"],
                "project_path": mem.get("project_path")
            }
            insights.append(insight)

        return insights

    async def suggest_claude_md_updates(
        self,
        min_confidence: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Generate suggestions for CLAUDE.md updates based on insights.

        Returns:
            List of suggested instructions to add to CLAUDE.md
        """
        # Get high-confidence insights that haven't been applied
        insights = await self.db.execute_query(
            """
            SELECT * FROM insights
            WHERE status = 'active'
            AND applied_to_claude_md = 0
            AND confidence >= ?
            ORDER BY impact_score DESC, confidence DESC
            LIMIT 20
            """,
            (min_confidence,)
        )

        if not insights:
            return []

        suggestions = []
        for insight in insights:
            insight_type = insight["insight_type"]
            title = insight["title"]
            desc = insight["description"]

            # Generate appropriate instruction based on type
            if insight_type == "recurring_error":
                instruction = f"- When encountering similar issues: {desc[:200]}"
                section = "Debugging & Errors"
            elif insight_type == "decision_pattern":
                instruction = f"- Standard approach: {desc[:200]}"
                section = "Development Patterns"
            elif insight_type == "correction_pattern":
                instruction = f"- Reminder: {desc[:200]}"
                section = "Important Notes"
            elif insight_type == "high_value_memory":
                instruction = f"- Reference: {desc[:200]}"
                section = "Quick Reference"
            else:
                instruction = f"- {desc[:200]}"
                section = "General"

            suggestions.append({
                "insight_id": insight["id"],
                "section": section,
                "instruction": instruction,
                "confidence": insight["confidence"],
                "impact_score": insight["impact_score"],
                "evidence_count": insight["evidence_count"]
            })

        return suggestions

    async def run_aggregation(
        self,
        days_back: int = 30
    ) -> Dict[str, Any]:
        """Run full aggregation pipeline.

        Returns:
            Summary of generated insights
        """
        results = {
            "error_patterns": 0,
            "decision_patterns": 0,
            "correction_patterns": 0,
            "high_value_memories": 0,
            "total_insights": 0
        }

        # Run each aggregation
        error_insights = await self.aggregate_error_patterns(days_back)
        for insight in error_insights:
            await self._store_insight(insight)
        results["error_patterns"] = len(error_insights)

        decision_insights = await self.aggregate_decision_patterns(days_back)
        for insight in decision_insights:
            await self._store_insight(insight)
        results["decision_patterns"] = len(decision_insights)

        correction_insights = await self.detect_correction_patterns(days_back)
        for insight in correction_insights:
            await self._store_insight(insight)
        results["correction_patterns"] = len(correction_insights)

        high_value = await self.identify_high_value_memories(days_back * 3)
        for insight in high_value:
            await self._store_insight(insight)
        results["high_value_memories"] = len(high_value)

        results["total_insights"] = sum([
            results["error_patterns"],
            results["decision_patterns"],
            results["correction_patterns"],
            results["high_value_memories"]
        ])

        return results

    async def _store_insight(self, insight: Dict[str, Any]) -> int:
        """Store an insight in the database."""
        # Generate embedding for the insight
        embedding = None
        if self.embeddings:
            text = f"{insight.get('title', '')} {insight.get('description', '')}"
            embedding = await self.embeddings.generate_embedding(text)

        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            INSERT INTO insights (
                insight_type, title, description, evidence_ids, evidence_count,
                source_sessions, confidence, impact_score, category,
                project_path, tech_context, embedding, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')
            """,
            (
                insight.get("insight_type"),
                insight.get("title"),
                insight.get("description"),
                insight.get("evidence_ids"),
                insight.get("evidence_count", 1),
                insight.get("source_sessions"),
                insight.get("confidence", 0.5),
                insight.get("impact_score", 5.0),
                insight.get("category"),
                insight.get("project_path"),
                insight.get("tech_context"),
                json.dumps(embedding) if embedding else None
            )
        )
        self.db.conn.commit()
        return cursor.lastrowid

    async def _cluster_by_embedding(
        self,
        items: List[Dict[str, Any]],
        threshold: float
    ) -> List[List[Dict[str, Any]]]:
        """Cluster items by embedding similarity.

        Simple greedy clustering algorithm.
        """
        if not items:
            return []

        clusters = []
        used = set()

        for i, item in enumerate(items):
            if i in used:
                continue

            cluster = [item]
            used.add(i)

            item_emb = self._parse_embedding(item.get("embedding"))
            if not item_emb:
                clusters.append(cluster)
                continue

            # Find similar items
            for j, other in enumerate(items[i+1:], start=i+1):
                if j in used:
                    continue

                other_emb = self._parse_embedding(other.get("embedding"))
                if not other_emb:
                    continue

                similarity = self._cosine_similarity(item_emb, other_emb)
                if similarity >= threshold:
                    cluster.append(other)
                    used.add(j)

            clusters.append(cluster)

        return [c for c in clusters if len(c) >= 1]

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

    def _summarize_cluster(self, cluster: List[Dict[str, Any]], cluster_type: str) -> str:
        """Generate a summary description for a cluster."""
        if not cluster:
            return ""

        # Use the first item as representative
        first = cluster[0]
        content = first.get("content", "")[:300]

        if cluster_type == "error":
            return f"Error pattern seen {len(cluster)} times: {content}"
        elif cluster_type == "decision":
            return f"Decision pattern applied {len(cluster)} times: {content}"
        else:
            return f"Pattern ({len(cluster)} occurrences): {content}"

    async def get_insights(
        self,
        insight_type: Optional[str] = None,
        project_path: Optional[str] = None,
        min_confidence: float = 0.0,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Retrieve stored insights.

        Args:
            insight_type: Filter by type (recurring_error, decision_pattern, etc.)
            project_path: Filter by project
            min_confidence: Minimum confidence threshold
            limit: Maximum results

        Returns:
            List of insights
        """
        query = """
            SELECT * FROM insights
            WHERE status = 'active'
            AND confidence >= ?
        """
        params = [min_confidence]

        if insight_type:
            query += " AND insight_type = ?"
            params.append(insight_type)

        if project_path:
            query += " AND (project_path = ? OR project_path IS NULL)"
            params.append(project_path)

        query += " ORDER BY impact_score DESC, confidence DESC LIMIT ?"
        params.append(limit)

        results = await self.db.execute_query(query, tuple(params))
        return [dict(r) for r in results] if results else []

    async def record_feedback(
        self,
        insight_id: int,
        helpful: bool,
        session_id: Optional[str] = None,
        comment: Optional[str] = None
    ) -> bool:
        """Record user feedback on an insight.

        Args:
            insight_id: The insight ID
            helpful: Whether the insight was helpful
            session_id: Current session
            comment: Optional feedback comment

        Returns:
            True if recorded successfully
        """
        cursor = self.db.conn.cursor()

        # Record feedback
        cursor.execute(
            """
            INSERT INTO insight_feedback (insight_id, session_id, feedback_type, helpful, comment)
            VALUES (?, ?, ?, ?, ?)
            """,
            (insight_id, session_id, "usage", 1 if helpful else 0, comment)
        )

        # Update insight validation counts
        if helpful:
            cursor.execute(
                """
                UPDATE insights
                SET validation_count = validation_count + 1,
                    confidence = MIN(0.99, confidence + 0.02),
                    last_validated_at = datetime('now'),
                    updated_at = datetime('now')
                WHERE id = ?
                """,
                (insight_id,)
            )
        else:
            cursor.execute(
                """
                UPDATE insights
                SET invalidation_count = invalidation_count + 1,
                    confidence = MAX(0.1, confidence - 0.05),
                    updated_at = datetime('now')
                WHERE id = ?
                """,
                (insight_id,)
            )

        self.db.conn.commit()
        return True

    async def mark_applied_to_claude_md(self, insight_id: int) -> bool:
        """Mark an insight as applied to CLAUDE.md."""
        cursor = self.db.conn.cursor()
        cursor.execute(
            """
            UPDATE insights
            SET applied_to_claude_md = 1,
                updated_at = datetime('now')
            WHERE id = ?
            """,
            (insight_id,)
        )
        self.db.conn.commit()
        return cursor.rowcount > 0


# Global instance
_insights: Optional[InsightsService] = None


def get_insights_service(db, embeddings) -> InsightsService:
    """Get the global insights service instance."""
    global _insights
    if _insights is None:
        _insights = InsightsService(db, embeddings)
    return _insights
