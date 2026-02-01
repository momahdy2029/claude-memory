"""Memory Curator Service - Autonomous graph exploration and maintenance.

The curator agent traverses the memory knowledge graph, finds duplicates,
suggests relationships, scores quality, and provides curated context.
"""
import logging
import json
import asyncio
from typing import Dict, Any, Optional, List, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class MemoryCurator:
    """
    Autonomous curator agent for memory graph maintenance.

    Capabilities:
    - Graph exploration (BFS/DFS traversal)
    - Duplicate detection (semantic similarity >0.92)
    - Relationship inference (suggest missing links)
    - Quality scoring (usage + connections + confidence)
    - Curated context generation
    - Scheduled maintenance
    """

    # Confidence thresholds for autonomous actions
    HIGH_CONFIDENCE = 0.9    # Auto-execute
    MEDIUM_CONFIDENCE = 0.7  # Suggest with one-click approval
    LOW_CONFIDENCE = 0.5     # Log for manual review only

    # Default configuration
    DEFAULT_CONFIG = {
        "auto_dedup_enabled": True,
        "auto_link_enabled": True,
        "dedup_threshold": 0.92,
        "maintenance_interval_hours": 24,
        "curator_active": True
    }

    def __init__(self, db, embeddings):
        """
        Initialize the curator with database and embedding services.

        Args:
            db: DatabaseService instance
            embeddings: EmbeddingService instance
        """
        self.db = db
        self.embeddings = embeddings
        self._running = False
        self._last_maintenance: Dict[str, datetime] = {}

    # ================================================================
    # GRAPH EXPLORATION
    # ================================================================

    async def explore_graph(
        self,
        start_node_id: int,
        max_depth: int = 3,
        mode: str = "bfs",
        relationship_filter: Optional[List[str]] = None,
        include_orphan_check: bool = True
    ) -> Dict[str, Any]:
        """
        Explore the memory graph from a starting node.

        Args:
            start_node_id: ID of the memory to start from
            max_depth: Maximum traversal depth
            mode: 'bfs' (breadth-first) or 'dfs' (depth-first)
            relationship_filter: Only follow these relationship types
            include_orphan_check: Check for orphaned nodes in the exploration

        Returns:
            Dict with explored nodes, edges, clusters, and insights
        """
        cursor = self.db.conn.cursor()

        # Verify start node exists
        cursor.execute("SELECT id, content, type FROM memories WHERE id = ?", (start_node_id,))
        start_node = cursor.fetchone()
        if not start_node:
            return {"error": f"Memory {start_node_id} not found"}

        visited: Set[int] = set()
        nodes: List[Dict] = []
        edges: List[Dict] = []
        depth_map: Dict[int, int] = {start_node_id: 0}

        # BFS/DFS exploration
        if mode == "bfs":
            queue = [start_node_id]
            while queue:
                current_id = queue.pop(0)
                if current_id in visited:
                    continue
                visited.add(current_id)

                current_depth = depth_map.get(current_id, 0)
                if current_depth >= max_depth:
                    continue

                # Get node info
                node_info = await self._get_node_info(current_id)
                if node_info:
                    node_info["depth"] = current_depth
                    nodes.append(node_info)

                # Get connected nodes
                neighbors = await self._get_neighbors(
                    current_id,
                    relationship_filter
                )

                for neighbor_id, edge_info in neighbors:
                    edges.append(edge_info)
                    if neighbor_id not in visited:
                        queue.append(neighbor_id)
                        if neighbor_id not in depth_map:
                            depth_map[neighbor_id] = current_depth + 1
        else:  # DFS
            stack = [start_node_id]
            while stack:
                current_id = stack.pop()
                if current_id in visited:
                    continue
                visited.add(current_id)

                current_depth = depth_map.get(current_id, 0)
                if current_depth >= max_depth:
                    continue

                node_info = await self._get_node_info(current_id)
                if node_info:
                    node_info["depth"] = current_depth
                    nodes.append(node_info)

                neighbors = await self._get_neighbors(
                    current_id,
                    relationship_filter
                )

                for neighbor_id, edge_info in neighbors:
                    edges.append(edge_info)
                    if neighbor_id not in visited:
                        stack.append(neighbor_id)
                        if neighbor_id not in depth_map:
                            depth_map[neighbor_id] = current_depth + 1

        # Identify clusters
        clusters = self._identify_clusters(nodes, edges)

        # Find orphans if requested
        orphans = []
        if include_orphan_check:
            orphans = await self.find_orphan_memories(limit=10)

        return {
            "start_node": start_node_id,
            "mode": mode,
            "max_depth": max_depth,
            "nodes_explored": len(nodes),
            "edges_found": len(edges),
            "nodes": nodes,
            "edges": edges,
            "clusters": clusters,
            "orphans_nearby": orphans[:5] if orphans else [],
            "exploration_insights": self._generate_exploration_insights(nodes, edges, clusters)
        }

    async def _get_node_info(self, memory_id: int) -> Optional[Dict]:
        """Get detailed info for a memory node."""
        cursor = self.db.conn.cursor()
        cursor.execute("""
            SELECT id, type, content, importance, confidence,
                   access_count, decay_factor, project_path, created_at
            FROM memories WHERE id = ?
        """, (memory_id,))
        row = cursor.fetchone()
        if not row:
            return None

        # Get relationship counts
        cursor.execute("""
            SELECT COUNT(*) as outgoing FROM memory_relationships WHERE source_id = ?
        """, (memory_id,))
        outgoing = cursor.fetchone()["outgoing"]

        cursor.execute("""
            SELECT COUNT(*) as incoming FROM memory_relationships WHERE target_id = ?
        """, (memory_id,))
        incoming = cursor.fetchone()["incoming"]

        return {
            "id": row["id"],
            "type": row["type"],
            "content": row["content"][:200] + "..." if len(row["content"]) > 200 else row["content"],
            "importance": row["importance"],
            "confidence": row["confidence"],
            "access_count": row["access_count"],
            "decay_factor": row["decay_factor"],
            "project_path": row["project_path"],
            "created_at": row["created_at"],
            "connection_count": outgoing + incoming,
            "outgoing_edges": outgoing,
            "incoming_edges": incoming
        }

    async def _get_neighbors(
        self,
        memory_id: int,
        relationship_filter: Optional[List[str]] = None
    ) -> List[Tuple[int, Dict]]:
        """Get all neighboring nodes and edge info."""
        cursor = self.db.conn.cursor()

        query = """
            SELECT target_id as neighbor_id, relationship, strength, 'outgoing' as direction
            FROM memory_relationships WHERE source_id = ?
            UNION ALL
            SELECT source_id as neighbor_id, relationship, strength, 'incoming' as direction
            FROM memory_relationships WHERE target_id = ?
        """
        cursor.execute(query, (memory_id, memory_id))

        neighbors = []
        for row in cursor.fetchall():
            if relationship_filter and row["relationship"] not in relationship_filter:
                continue

            edge_info = {
                "source": memory_id if row["direction"] == "outgoing" else row["neighbor_id"],
                "target": row["neighbor_id"] if row["direction"] == "outgoing" else memory_id,
                "relationship": row["relationship"],
                "strength": row["strength"],
                "direction": row["direction"]
            }
            neighbors.append((row["neighbor_id"], edge_info))

        return neighbors

    def _identify_clusters(self, nodes: List[Dict], edges: List[Dict]) -> List[Dict]:
        """Identify clusters of tightly connected nodes."""
        if not nodes:
            return []

        # Build adjacency for clustering
        adjacency = defaultdict(set)
        for edge in edges:
            adjacency[edge["source"]].add(edge["target"])
            adjacency[edge["target"]].add(edge["source"])

        # Simple connected component analysis
        visited = set()
        clusters = []

        for node in nodes:
            node_id = node["id"]
            if node_id in visited:
                continue

            # BFS to find component
            component = []
            queue = [node_id]
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                component.append(current)
                for neighbor in adjacency[current]:
                    if neighbor not in visited:
                        queue.append(neighbor)

            if len(component) > 1:
                # Determine cluster type based on node types
                node_types = defaultdict(int)
                for nid in component:
                    for n in nodes:
                        if n["id"] == nid:
                            node_types[n["type"]] += 1
                            break

                clusters.append({
                    "node_ids": component,
                    "size": len(component),
                    "dominant_type": max(node_types, key=node_types.get) if node_types else "mixed",
                    "type_distribution": dict(node_types)
                })

        return sorted(clusters, key=lambda c: c["size"], reverse=True)

    def _generate_exploration_insights(
        self,
        nodes: List[Dict],
        edges: List[Dict],
        clusters: List[Dict]
    ) -> List[str]:
        """Generate insights from the exploration."""
        insights = []

        if not nodes:
            return ["No nodes found in exploration"]

        # Type distribution
        type_counts = defaultdict(int)
        for node in nodes:
            type_counts[node["type"]] += 1

        dominant = max(type_counts, key=type_counts.get)
        insights.append(f"Dominant memory type: {dominant} ({type_counts[dominant]}/{len(nodes)})")

        # Connection density
        if nodes:
            avg_connections = sum(n.get("connection_count", 0) for n in nodes) / len(nodes)
            if avg_connections < 1:
                insights.append("Low connectivity: Consider adding more relationships")
            elif avg_connections > 5:
                insights.append("High connectivity: Knowledge graph is well-connected")

        # Cluster analysis
        if clusters:
            largest = clusters[0]
            insights.append(f"Largest cluster: {largest['size']} nodes ({largest['dominant_type']})")

        # Quality indicators
        low_confidence = [n for n in nodes if n.get("confidence", 0.5) < 0.3]
        if low_confidence:
            insights.append(f"{len(low_confidence)} nodes with low confidence need review")

        high_importance = [n for n in nodes if n.get("importance", 5) >= 8]
        if high_importance:
            insights.append(f"{len(high_importance)} high-importance nodes in this subgraph")

        return insights

    # ================================================================
    # DUPLICATE DETECTION
    # ================================================================

    async def find_duplicates(
        self,
        project_path: Optional[str] = None,
        similarity_threshold: float = 0.92,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Find semantically similar (duplicate) memories.

        Args:
            project_path: Optional project filter
            similarity_threshold: Minimum similarity to consider duplicates (default 0.92)
            limit: Maximum number of duplicate pairs to return

        Returns:
            Dict with duplicate clusters and merge suggestions
        """
        cursor = self.db.conn.cursor()

        # Get memories with embeddings
        if project_path:
            from services.database import normalize_path
            normalized = normalize_path(project_path)
            cursor.execute("""
                SELECT id, content, type, importance, confidence, embedding, created_at
                FROM memories
                WHERE embedding IS NOT NULL AND project_path = ?
                ORDER BY created_at DESC
                LIMIT 500
            """, (normalized,))
        else:
            cursor.execute("""
                SELECT id, content, type, importance, confidence, embedding, created_at
                FROM memories
                WHERE embedding IS NOT NULL
                ORDER BY created_at DESC
                LIMIT 500
            """)

        memories = cursor.fetchall()
        if len(memories) < 2:
            return {"duplicate_clusters": [], "total_memories_checked": len(memories)}

        # Parse embeddings
        memory_data = []
        for mem in memories:
            try:
                embedding = json.loads(mem["embedding"])
                memory_data.append({
                    "id": mem["id"],
                    "content": mem["content"],
                    "type": mem["type"],
                    "importance": mem["importance"],
                    "confidence": mem["confidence"],
                    "embedding": embedding,
                    "created_at": mem["created_at"]
                })
            except (json.JSONDecodeError, TypeError):
                continue

        # Find duplicate pairs
        import numpy as np
        duplicate_pairs = []
        checked_pairs = set()

        for i, mem1 in enumerate(memory_data):
            for j, mem2 in enumerate(memory_data):
                if i >= j:
                    continue

                pair_key = (min(mem1["id"], mem2["id"]), max(mem1["id"], mem2["id"]))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                # Calculate cosine similarity
                try:
                    vec1 = np.array(mem1["embedding"])
                    vec2 = np.array(mem2["embedding"])
                    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

                    if similarity >= similarity_threshold:
                        duplicate_pairs.append({
                            "memory1": {
                                "id": mem1["id"],
                                "content": mem1["content"][:150],
                                "type": mem1["type"],
                                "importance": mem1["importance"],
                                "confidence": mem1["confidence"],
                                "created_at": mem1["created_at"]
                            },
                            "memory2": {
                                "id": mem2["id"],
                                "content": mem2["content"][:150],
                                "type": mem2["type"],
                                "importance": mem2["importance"],
                                "confidence": mem2["confidence"],
                                "created_at": mem2["created_at"]
                            },
                            "similarity": float(similarity),
                            "merge_recommendation": self._get_merge_recommendation(mem1, mem2, similarity)
                        })
                except Exception as e:
                    logger.debug(f"Error calculating similarity: {e}")
                    continue

        # Sort by similarity and limit
        duplicate_pairs.sort(key=lambda x: x["similarity"], reverse=True)
        duplicate_pairs = duplicate_pairs[:limit]

        # Cluster duplicates (transitive grouping)
        clusters = self._cluster_duplicates(duplicate_pairs)

        return {
            "duplicate_clusters": clusters,
            "duplicate_pairs": duplicate_pairs,
            "total_memories_checked": len(memory_data),
            "duplicates_found": len(duplicate_pairs),
            "threshold_used": similarity_threshold,
            "auto_merge_candidates": [
                p for p in duplicate_pairs
                if p["merge_recommendation"]["confidence"] >= self.HIGH_CONFIDENCE
            ]
        }

    def _get_merge_recommendation(
        self,
        mem1: Dict,
        mem2: Dict,
        similarity: float
    ) -> Dict[str, Any]:
        """Determine which memory to keep in a merge."""
        # Scoring: higher is better to keep
        score1 = 0
        score2 = 0

        # Prefer higher importance
        score1 += mem1["importance"] * 2
        score2 += mem2["importance"] * 2

        # Prefer higher confidence
        score1 += mem1["confidence"] * 10
        score2 += mem2["confidence"] * 10

        # Prefer longer content (more detail)
        score1 += min(len(mem1["content"]) / 100, 5)
        score2 += min(len(mem2["content"]) / 100, 5)

        # Prefer newer for decisions, older for established patterns
        if mem1["type"] == "decision":
            # Newer decisions are more relevant
            score1 += 3 if mem1["created_at"] > mem2["created_at"] else 0
            score2 += 3 if mem2["created_at"] > mem1["created_at"] else 0
        else:
            # Older patterns are more established
            score1 += 2 if mem1["created_at"] < mem2["created_at"] else 0
            score2 += 2 if mem2["created_at"] < mem1["created_at"] else 0

        keep_id = mem1["id"] if score1 >= score2 else mem2["id"]
        remove_id = mem2["id"] if score1 >= score2 else mem1["id"]

        # Confidence in recommendation
        score_diff = abs(score1 - score2)
        if score_diff > 10 and similarity > 0.95:
            confidence = self.HIGH_CONFIDENCE
        elif score_diff > 5 and similarity > 0.93:
            confidence = self.MEDIUM_CONFIDENCE
        else:
            confidence = self.LOW_CONFIDENCE

        return {
            "keep": keep_id,
            "remove": remove_id,
            "confidence": confidence,
            "reason": f"Score {keep_id}={max(score1,score2):.1f} vs {remove_id}={min(score1,score2):.1f}"
        }

    def _cluster_duplicates(self, pairs: List[Dict]) -> List[Dict]:
        """Cluster duplicate pairs into groups."""
        if not pairs:
            return []

        # Build union-find
        parent = {}

        def find(x):
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Union all pairs
        for pair in pairs:
            union(pair["memory1"]["id"], pair["memory2"]["id"])

        # Group by root
        clusters_map = defaultdict(list)
        all_ids = set()
        for pair in pairs:
            all_ids.add(pair["memory1"]["id"])
            all_ids.add(pair["memory2"]["id"])

        for mem_id in all_ids:
            root = find(mem_id)
            clusters_map[root].append(mem_id)

        # Build cluster objects
        clusters = []
        for root, members in clusters_map.items():
            if len(members) > 1:
                # Find the best candidate to keep
                best_id = None
                best_score = -1
                for pair in pairs:
                    if pair["memory1"]["id"] in members:
                        rec = pair["merge_recommendation"]
                        if rec["keep"] in members and rec["confidence"] > best_score:
                            best_id = rec["keep"]
                            best_score = rec["confidence"]

                clusters.append({
                    "member_ids": sorted(members),
                    "size": len(members),
                    "recommended_keep": best_id,
                    "merge_confidence": best_score
                })

        return sorted(clusters, key=lambda c: c["size"], reverse=True)

    # ================================================================
    # RELATIONSHIP INFERENCE
    # ================================================================

    async def suggest_relationships(
        self,
        memory_id: Optional[int] = None,
        project_path: Optional[str] = None,
        similarity_threshold: float = 0.7,
        limit: int = 20
    ) -> Dict[str, Any]:
        """
        Suggest missing relationships between memories.

        Uses semantic similarity and content analysis to infer
        relationships that should exist but don't.

        Args:
            memory_id: Optional specific memory to find links for
            project_path: Optional project filter
            similarity_threshold: Minimum similarity for suggestions
            limit: Maximum suggestions to return

        Returns:
            Dict with suggested relationships
        """
        cursor = self.db.conn.cursor()

        suggestions = []

        if memory_id:
            # Find relationships for a specific memory
            cursor.execute("""
                SELECT id, content, type, embedding FROM memories WHERE id = ?
            """, (memory_id,))
            source = cursor.fetchone()
            if not source or not source["embedding"]:
                return {"suggestions": [], "error": "Memory not found or has no embedding"}

            source_embedding = json.loads(source["embedding"])

            # Get existing relationships
            cursor.execute("""
                SELECT target_id FROM memory_relationships WHERE source_id = ?
                UNION
                SELECT source_id FROM memory_relationships WHERE target_id = ?
            """, (memory_id, memory_id))
            existing = {row[0] for row in cursor.fetchall()}
            existing.add(memory_id)

            # Find similar unconnected memories
            cursor.execute("""
                SELECT id, content, type, embedding, importance
                FROM memories
                WHERE embedding IS NOT NULL AND id NOT IN ({})
                LIMIT 200
            """.format(','.join('?' * len(existing))), tuple(existing))

            import numpy as np
            source_vec = np.array(source_embedding)

            for row in cursor.fetchall():
                try:
                    target_vec = np.array(json.loads(row["embedding"]))
                    similarity = np.dot(source_vec, target_vec) / (
                        np.linalg.norm(source_vec) * np.linalg.norm(target_vec)
                    )

                    if similarity >= similarity_threshold:
                        rel_type = self._infer_relationship_type(
                            source["type"], source["content"],
                            row["type"], row["content"]
                        )

                        suggestions.append({
                            "source_id": memory_id,
                            "target_id": row["id"],
                            "relationship": rel_type,
                            "similarity": float(similarity),
                            "confidence": self._calculate_link_confidence(
                                similarity, source["type"], row["type"]
                            ),
                            "source_preview": source["content"][:100],
                            "target_preview": row["content"][:100]
                        })
                except Exception as e:
                    logger.debug(f"Error processing memory {row['id']}: {e}")
                    continue
        else:
            # Find suggestions across the project
            if project_path:
                from services.database import normalize_path
                normalized = normalize_path(project_path)
                cursor.execute("""
                    SELECT id, content, type, embedding, importance
                    FROM memories
                    WHERE embedding IS NOT NULL AND project_path = ?
                    ORDER BY importance DESC
                    LIMIT 100
                """, (normalized,))
            else:
                cursor.execute("""
                    SELECT id, content, type, embedding, importance
                    FROM memories
                    WHERE embedding IS NOT NULL
                    ORDER BY importance DESC
                    LIMIT 100
                """)

            memories = cursor.fetchall()

            # Get all existing relationships
            cursor.execute("SELECT source_id, target_id FROM memory_relationships")
            existing_pairs = {(row[0], row[1]) for row in cursor.fetchall()}

            import numpy as np

            # Check pairs for potential relationships
            for i, mem1 in enumerate(memories):
                if len(suggestions) >= limit:
                    break

                for mem2 in memories[i+1:]:
                    if len(suggestions) >= limit:
                        break

                    pair = (min(mem1["id"], mem2["id"]), max(mem1["id"], mem2["id"]))
                    if pair in existing_pairs or (pair[1], pair[0]) in existing_pairs:
                        continue

                    try:
                        vec1 = np.array(json.loads(mem1["embedding"]))
                        vec2 = np.array(json.loads(mem2["embedding"]))
                        similarity = np.dot(vec1, vec2) / (
                            np.linalg.norm(vec1) * np.linalg.norm(vec2)
                        )

                        if similarity >= similarity_threshold:
                            rel_type = self._infer_relationship_type(
                                mem1["type"], mem1["content"],
                                mem2["type"], mem2["content"]
                            )

                            suggestions.append({
                                "source_id": mem1["id"],
                                "target_id": mem2["id"],
                                "relationship": rel_type,
                                "similarity": float(similarity),
                                "confidence": self._calculate_link_confidence(
                                    similarity, mem1["type"], mem2["type"]
                                ),
                                "source_preview": mem1["content"][:100],
                                "target_preview": mem2["content"][:100]
                            })
                    except Exception:
                        continue

        # Sort by confidence
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        suggestions = suggestions[:limit]

        return {
            "suggestions": suggestions,
            "total_found": len(suggestions),
            "auto_apply_candidates": [
                s for s in suggestions
                if s["confidence"] >= self.HIGH_CONFIDENCE
            ]
        }

    def _infer_relationship_type(
        self,
        type1: str, content1: str,
        type2: str, content2: str
    ) -> str:
        """Infer the most likely relationship type between two memories."""
        content1_lower = content1.lower()
        content2_lower = content2.lower()

        # Error + fix pattern
        if type1 == "error" and type2 in ["code", "decision"]:
            if any(w in content2_lower for w in ["fix", "solve", "resolve", "solution"]):
                return "fixes"
        if type2 == "error" and type1 in ["code", "decision"]:
            if any(w in content1_lower for w in ["fix", "solve", "resolve", "solution"]):
                return "fixes"

        # Cause-effect pattern
        if any(w in content1_lower for w in ["because", "caused", "led to", "resulted"]):
            return "caused_by"
        if any(w in content2_lower for w in ["because", "caused", "led to", "resulted"]):
            return "caused_by"

        # Contradiction pattern
        if any(w in content1_lower for w in ["but", "however", "instead", "contrary"]):
            return "contradicts"
        if any(w in content2_lower for w in ["but", "however", "instead", "contrary"]):
            return "contradicts"

        # Support pattern
        if type1 == type2 == "decision":
            return "supports"

        # Default to related
        return "related"

    def _calculate_link_confidence(
        self,
        similarity: float,
        type1: str,
        type2: str
    ) -> float:
        """Calculate confidence score for a suggested link."""
        base = similarity

        # Boost for complementary types
        complementary = {
            ("error", "code"): 0.1,
            ("error", "decision"): 0.1,
            ("decision", "decision"): 0.05,
            ("code", "code"): 0.05,
        }

        pair = (type1, type2) if type1 <= type2 else (type2, type1)
        boost = complementary.get(pair, 0)

        return min(base + boost, 1.0)

    # ================================================================
    # QUALITY SCORING
    # ================================================================

    async def score_quality(
        self,
        memory_id: Optional[int] = None,
        project_path: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Calculate quality scores for memories.

        Quality = f(usage, connections, confidence, age_decay)

        Args:
            memory_id: Optional specific memory to score
            project_path: Optional project filter
            limit: Maximum memories to score

        Returns:
            Dict with quality scores and insights
        """
        cursor = self.db.conn.cursor()

        if memory_id:
            cursor.execute("""
                SELECT id, content, type, importance, confidence,
                       access_count, decay_factor, created_at
                FROM memories WHERE id = ?
            """, (memory_id,))
            memories = cursor.fetchall()
        elif project_path:
            from services.database import normalize_path
            normalized = normalize_path(project_path)
            cursor.execute("""
                SELECT id, content, type, importance, confidence,
                       access_count, decay_factor, created_at
                FROM memories WHERE project_path = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (normalized, limit))
            memories = cursor.fetchall()
        else:
            cursor.execute("""
                SELECT id, content, type, importance, confidence,
                       access_count, decay_factor, created_at
                FROM memories
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))
            memories = cursor.fetchall()

        scores = []
        for mem in memories:
            # Get connection count
            cursor.execute("""
                SELECT COUNT(*) as count FROM memory_relationships
                WHERE source_id = ? OR target_id = ?
            """, (mem["id"], mem["id"]))
            connections = cursor.fetchone()["count"]

            # Calculate quality score
            quality = self._calculate_quality_score(
                importance=mem["importance"],
                confidence=mem["confidence"],
                access_count=mem["access_count"],
                decay_factor=mem["decay_factor"],
                connections=connections
            )

            scores.append({
                "id": mem["id"],
                "type": mem["type"],
                "content_preview": mem["content"][:100],
                "quality_score": quality,
                "components": {
                    "importance": mem["importance"],
                    "confidence": mem["confidence"],
                    "usage": mem["access_count"],
                    "decay": mem["decay_factor"],
                    "connections": connections
                },
                "needs_attention": quality < 0.3,
                "is_high_quality": quality > 0.7
            })

        scores.sort(key=lambda x: x["quality_score"], reverse=True)

        # Generate insights
        low_quality = [s for s in scores if s["quality_score"] < 0.3]
        high_quality = [s for s in scores if s["quality_score"] > 0.7]
        avg_quality = sum(s["quality_score"] for s in scores) / len(scores) if scores else 0

        return {
            "scores": scores,
            "summary": {
                "total_scored": len(scores),
                "average_quality": round(avg_quality, 3),
                "high_quality_count": len(high_quality),
                "needs_attention_count": len(low_quality)
            },
            "needs_attention": low_quality[:10],
            "top_quality": high_quality[:10]
        }

    def _calculate_quality_score(
        self,
        importance: int,
        confidence: float,
        access_count: int,
        decay_factor: float,
        connections: int
    ) -> float:
        """Calculate overall quality score (0-1)."""
        # Normalize components
        importance_norm = (importance or 5) / 10  # 0-1
        confidence_norm = confidence or 0.5  # Already 0-1
        usage_norm = min((access_count or 0) / 20, 1)  # Cap at 20 uses
        decay_norm = decay_factor or 1.0  # Already 0-1
        connection_norm = min(connections / 10, 1)  # Cap at 10 connections

        # Weighted average
        weights = {
            "importance": 0.25,
            "confidence": 0.25,
            "usage": 0.15,
            "decay": 0.15,
            "connections": 0.20
        }

        score = (
            importance_norm * weights["importance"] +
            confidence_norm * weights["confidence"] +
            usage_norm * weights["usage"] +
            decay_norm * weights["decay"] +
            connection_norm * weights["connections"]
        )

        return round(score, 3)

    # ================================================================
    # ORPHAN DETECTION
    # ================================================================

    async def find_orphan_memories(
        self,
        project_path: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict]:
        """Find memories with no relationships."""
        cursor = self.db.conn.cursor()

        if project_path:
            from services.database import normalize_path
            normalized = normalize_path(project_path)
            cursor.execute("""
                SELECT m.id, m.content, m.type, m.importance, m.confidence, m.created_at
                FROM memories m
                LEFT JOIN memory_relationships mr1 ON m.id = mr1.source_id
                LEFT JOIN memory_relationships mr2 ON m.id = mr2.target_id
                WHERE mr1.id IS NULL AND mr2.id IS NULL AND m.project_path = ?
                ORDER BY m.importance DESC, m.created_at DESC
                LIMIT ?
            """, (normalized, limit))
        else:
            cursor.execute("""
                SELECT m.id, m.content, m.type, m.importance, m.confidence, m.created_at
                FROM memories m
                LEFT JOIN memory_relationships mr1 ON m.id = mr1.source_id
                LEFT JOIN memory_relationships mr2 ON m.id = mr2.target_id
                WHERE mr1.id IS NULL AND mr2.id IS NULL
                ORDER BY m.importance DESC, m.created_at DESC
                LIMIT ?
            """, (limit,))

        orphans = []
        for row in cursor.fetchall():
            orphans.append({
                "id": row["id"],
                "content": row["content"][:150],
                "type": row["type"],
                "importance": row["importance"],
                "confidence": row["confidence"],
                "created_at": row["created_at"]
            })

        return orphans

    # ================================================================
    # CURATED CONTEXT GENERATION
    # ================================================================

    async def generate_summary(
        self,
        query: str,
        project_path: Optional[str] = None,
        max_memories: int = 10,
        include_graph: bool = True
    ) -> Dict[str, Any]:
        """
        Generate curated context summary for a query.

        This is what gets injected into the main Claude's context
        via the grounding hook.

        Args:
            query: The topic/query to generate context for
            project_path: Optional project filter
            max_memories: Maximum memories to include
            include_graph: Include relationship graph context

        Returns:
            Dict with curated context summary
        """
        # Search for relevant memories
        from skills.search import semantic_search
        results = await semantic_search(
            db=self.db,
            embeddings=self.embeddings,
            query=query,
            limit=max_memories,
            project_path=project_path,
            threshold=0.5
        )

        memories = results.get("results", [])

        if not memories:
            return {
                "query": query,
                "context": "No relevant memories found.",
                "memories": [],
                "graph_context": None
            }

        # Build context sections
        sections = []

        # Group by type
        by_type = defaultdict(list)
        for mem in memories:
            by_type[mem.get("type", "chunk")].append(mem)

        # Decisions first (most important for context)
        if by_type.get("decision"):
            sections.append("**Key Decisions:**")
            for mem in by_type["decision"][:3]:
                sections.append(f"- {mem['content'][:200]}")

        # Errors and fixes
        if by_type.get("error"):
            sections.append("\n**Known Issues:**")
            for mem in by_type["error"][:3]:
                sections.append(f"- {mem['content'][:200]}")

        # Code patterns
        if by_type.get("code"):
            sections.append("\n**Code Patterns:**")
            for mem in by_type["code"][:3]:
                sections.append(f"- {mem['content'][:200]}")

        # Other relevant
        other = [m for t, mems in by_type.items()
                 for m in mems if t not in ["decision", "error", "code"]]
        if other:
            sections.append("\n**Related Context:**")
            for mem in other[:3]:
                sections.append(f"- {mem['content'][:200]}")

        # Build graph context if requested
        graph_context = None
        if include_graph and memories:
            graph_context = await self._build_graph_context(memories)

        # Check for pending curator items
        pending = await self._get_pending_reviews(project_path)

        return {
            "query": query,
            "context": "\n".join(sections),
            "memories": [
                {"id": m["id"], "type": m.get("type"), "relevance": m.get("relevance", 0)}
                for m in memories
            ],
            "graph_context": graph_context,
            "pending_reviews": pending,
            "generated_at": datetime.now().isoformat()
        }

    async def _build_graph_context(self, memories: List[Dict]) -> Dict[str, Any]:
        """Build graph relationship context for memories."""
        memory_ids = [m["id"] for m in memories if m.get("id")]
        if not memory_ids:
            return None

        cursor = self.db.conn.cursor()

        # Get relationships between these memories
        placeholders = ','.join('?' * len(memory_ids))
        cursor.execute(f"""
            SELECT source_id, target_id, relationship, strength
            FROM memory_relationships
            WHERE source_id IN ({placeholders}) OR target_id IN ({placeholders})
        """, memory_ids + memory_ids)

        edges = []
        for row in cursor.fetchall():
            edges.append({
                "source": row["source_id"],
                "target": row["target_id"],
                "type": row["relationship"],
                "strength": row["strength"]
            })

        # Format as readable context
        if not edges:
            return {"edges": [], "summary": "No relationships between these memories"}

        relationship_summary = []
        for edge in edges[:10]:
            relationship_summary.append(
                f"Memory #{edge['source']} {edge['type']} Memory #{edge['target']}"
            )

        return {
            "edges": edges,
            "summary": "; ".join(relationship_summary),
            "edge_count": len(edges)
        }

    async def _get_pending_reviews(self, project_path: Optional[str] = None) -> Dict[str, Any]:
        """Get pending curator review items."""
        # Check for duplicates
        duplicates = await self.find_duplicates(
            project_path=project_path,
            similarity_threshold=0.92,
            limit=5
        )

        # Check for suggested links
        suggestions = await self.suggest_relationships(
            project_path=project_path,
            similarity_threshold=0.8,
            limit=5
        )

        # Check for orphans
        orphans = await self.find_orphan_memories(
            project_path=project_path,
            limit=5
        )

        return {
            "duplicate_clusters": len(duplicates.get("duplicate_clusters", [])),
            "suggested_links": len(suggestions.get("suggestions", [])),
            "orphan_memories": len(orphans),
            "total_pending": (
                len(duplicates.get("duplicate_clusters", [])) +
                len(suggestions.get("suggestions", [])) +
                len(orphans)
            )
        }

    # ================================================================
    # MERGE OPERATIONS
    # ================================================================

    async def merge_memories(
        self,
        keep_id: int,
        remove_ids: List[int],
        merge_content: bool = False
    ) -> Dict[str, Any]:
        """
        Merge duplicate memories into one.

        Args:
            keep_id: Memory ID to keep
            remove_ids: Memory IDs to merge into keep_id
            merge_content: If True, append removed content to kept memory

        Returns:
            Dict with merge result
        """
        cursor = self.db.conn.cursor()

        # Verify keep memory exists
        cursor.execute("SELECT * FROM memories WHERE id = ?", (keep_id,))
        keep_memory = cursor.fetchone()
        if not keep_memory:
            return {"error": f"Memory {keep_id} not found"}

        merged_count = 0
        merged_relationships = 0

        for remove_id in remove_ids:
            if remove_id == keep_id:
                continue

            cursor.execute("SELECT * FROM memories WHERE id = ?", (remove_id,))
            remove_memory = cursor.fetchone()
            if not remove_memory:
                continue

            # Transfer relationships
            # Update outgoing relationships
            cursor.execute("""
                UPDATE OR IGNORE memory_relationships
                SET source_id = ?
                WHERE source_id = ?
            """, (keep_id, remove_id))
            merged_relationships += cursor.rowcount

            # Update incoming relationships
            cursor.execute("""
                UPDATE OR IGNORE memory_relationships
                SET target_id = ?
                WHERE target_id = ?
            """, (keep_id, remove_id))
            merged_relationships += cursor.rowcount

            # Delete duplicate relationships
            cursor.execute("""
                DELETE FROM memory_relationships
                WHERE source_id = ? OR target_id = ?
            """, (remove_id, remove_id))

            # Optionally merge content
            if merge_content:
                cursor.execute("""
                    UPDATE memories
                    SET content = content || '\n\n[Merged from #' || ? || ']: ' || ?
                    WHERE id = ?
                """, (remove_id, remove_memory["content"], keep_id))

            # Archive the removed memory
            cursor.execute("""
                INSERT INTO memory_archive
                (original_id, type, content, embedding, project_path, session_id,
                 importance, access_count, decay_factor, metadata, archive_reason)
                SELECT id, type, content, embedding, project_path, session_id,
                       importance, access_count, decay_factor, metadata, 'merged'
                FROM memories WHERE id = ?
            """, (remove_id,))

            # Delete the memory
            cursor.execute("DELETE FROM memories WHERE id = ?", (remove_id,))
            merged_count += 1

        self.db.conn.commit()

        # Update importance if we merged several
        if merged_count > 0:
            new_importance = min(keep_memory["importance"] + merged_count, 10)
            cursor.execute("""
                UPDATE memories SET importance = ? WHERE id = ?
            """, (new_importance, keep_id))
            self.db.conn.commit()

        return {
            "success": True,
            "kept_id": keep_id,
            "merged_count": merged_count,
            "relationships_transferred": merged_relationships,
            "new_importance": min(keep_memory["importance"] + merged_count, 10)
        }

    # ================================================================
    # MAINTENANCE TASKS
    # ================================================================

    async def run_maintenance(
        self,
        project_path: Optional[str] = None,
        tasks: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run curator maintenance tasks.

        Args:
            project_path: Optional project filter
            tasks: Specific tasks to run, or None for all
                   Options: dedup, orphans, links, decay, quality

        Returns:
            Dict with maintenance report
        """
        all_tasks = ["dedup", "orphans", "links", "decay", "quality"]
        tasks_to_run = tasks or all_tasks

        report = {
            "started_at": datetime.now().isoformat(),
            "project_path": project_path,
            "tasks_run": [],
            "findings": {},
            "actions_taken": {},
            "recommendations": []
        }

        # Get config
        config = await self.get_config(project_path)

        if "dedup" in tasks_to_run and config.get("auto_dedup_enabled", True):
            duplicates = await self.find_duplicates(
                project_path=project_path,
                similarity_threshold=config.get("dedup_threshold", 0.92)
            )
            report["findings"]["duplicates"] = duplicates.get("duplicates_found", 0)
            report["tasks_run"].append("dedup")

            # Auto-merge high-confidence duplicates
            auto_merge = duplicates.get("auto_merge_candidates", [])
            if auto_merge:
                for pair in auto_merge[:5]:  # Limit auto-merges
                    rec = pair["merge_recommendation"]
                    await self.merge_memories(
                        keep_id=rec["keep"],
                        remove_ids=[rec["remove"]]
                    )
                report["actions_taken"]["auto_merged"] = len(auto_merge[:5])

        if "orphans" in tasks_to_run:
            orphans = await self.find_orphan_memories(project_path=project_path)
            report["findings"]["orphans"] = len(orphans)
            report["tasks_run"].append("orphans")

            if orphans:
                report["recommendations"].append(
                    f"Found {len(orphans)} orphan memories - consider linking or archiving"
                )

        if "links" in tasks_to_run and config.get("auto_link_enabled", True):
            suggestions = await self.suggest_relationships(
                project_path=project_path,
                similarity_threshold=0.75
            )
            report["findings"]["suggested_links"] = len(suggestions.get("suggestions", []))
            report["tasks_run"].append("links")

            # Auto-apply high-confidence links
            auto_links = suggestions.get("auto_apply_candidates", [])
            if auto_links:
                for link in auto_links[:10]:
                    await self.db.create_relationship(
                        source_id=link["source_id"],
                        target_id=link["target_id"],
                        relationship=link["relationship"],
                        strength=link["similarity"]
                    )
                report["actions_taken"]["auto_linked"] = len(auto_links[:10])

        if "quality" in tasks_to_run:
            quality = await self.score_quality(project_path=project_path)
            report["findings"]["quality_summary"] = quality.get("summary", {})
            report["tasks_run"].append("quality")

            needs_attention = quality.get("needs_attention", [])
            if needs_attention:
                report["recommendations"].append(
                    f"{len(needs_attention)} memories need attention (low quality score)"
                )

        if "decay" in tasks_to_run:
            # Apply confidence decay to unused memories
            decayed = await self._apply_confidence_decay(project_path)
            report["actions_taken"]["memories_decayed"] = decayed
            report["tasks_run"].append("decay")

        report["completed_at"] = datetime.now().isoformat()

        # Save report
        await self._save_report(report, project_path)

        return report

    async def _apply_confidence_decay(
        self,
        project_path: Optional[str] = None,
        decay_rate: float = 0.95
    ) -> int:
        """Apply decay to memories not accessed recently."""
        cursor = self.db.conn.cursor()

        # Decay memories not accessed in the last 30 days
        cutoff = (datetime.now() - timedelta(days=30)).isoformat()

        if project_path:
            from services.database import normalize_path
            normalized = normalize_path(project_path)
            cursor.execute("""
                UPDATE memories
                SET decay_factor = decay_factor * ?,
                    confidence = confidence * ?
                WHERE (last_accessed IS NULL OR last_accessed < ?)
                AND project_path = ?
                AND decay_factor > 0.1
            """, (decay_rate, decay_rate, cutoff, normalized))
        else:
            cursor.execute("""
                UPDATE memories
                SET decay_factor = decay_factor * ?,
                    confidence = confidence * ?
                WHERE (last_accessed IS NULL OR last_accessed < ?)
                AND decay_factor > 0.1
            """, (decay_rate, decay_rate, cutoff))

        decayed = cursor.rowcount
        self.db.conn.commit()
        return decayed

    async def _save_report(self, report: Dict, project_path: Optional[str] = None):
        """Save maintenance report to database."""
        cursor = self.db.conn.cursor()

        from services.database import normalize_path
        normalized = normalize_path(project_path) if project_path else None

        cursor.execute("""
            INSERT INTO curator_reports
            (project_path, report_type, summary, findings, actions_taken, recommendations)
            VALUES (?, 'maintenance', ?, ?, ?, ?)
        """, (
            normalized,
            f"Ran tasks: {', '.join(report.get('tasks_run', []))}",
            json.dumps(report.get("findings", {})),
            json.dumps(report.get("actions_taken", {})),
            json.dumps(report.get("recommendations", []))
        ))
        self.db.conn.commit()

    # ================================================================
    # CONFIGURATION
    # ================================================================

    async def get_config(self, project_path: Optional[str] = None) -> Dict[str, Any]:
        """Get curator configuration for a project."""
        cursor = self.db.conn.cursor()

        if project_path:
            from services.database import normalize_path
            normalized = normalize_path(project_path)
            cursor.execute("""
                SELECT * FROM curator_config WHERE project_path = ?
            """, (normalized,))
            row = cursor.fetchone()
            if row:
                return dict(row)

        return self.DEFAULT_CONFIG.copy()

    async def update_config(
        self,
        project_path: str,
        **config_updates
    ) -> Dict[str, Any]:
        """Update curator configuration for a project."""
        cursor = self.db.conn.cursor()

        from services.database import normalize_path
        normalized = normalize_path(project_path)

        # Get existing or default
        existing = await self.get_config(project_path)
        existing.update(config_updates)

        cursor.execute("""
            INSERT INTO curator_config
            (project_path, auto_dedup_enabled, auto_link_enabled, dedup_threshold,
             maintenance_interval_hours, curator_active)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(project_path) DO UPDATE SET
                auto_dedup_enabled = excluded.auto_dedup_enabled,
                auto_link_enabled = excluded.auto_link_enabled,
                dedup_threshold = excluded.dedup_threshold,
                maintenance_interval_hours = excluded.maintenance_interval_hours,
                curator_active = excluded.curator_active
        """, (
            normalized,
            existing.get("auto_dedup_enabled", True),
            existing.get("auto_link_enabled", True),
            existing.get("dedup_threshold", 0.92),
            existing.get("maintenance_interval_hours", 24),
            existing.get("curator_active", True)
        ))
        self.db.conn.commit()

        return existing

    async def get_latest_report(
        self,
        project_path: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get the latest curator report."""
        cursor = self.db.conn.cursor()

        if project_path:
            from services.database import normalize_path
            normalized = normalize_path(project_path)
            cursor.execute("""
                SELECT * FROM curator_reports
                WHERE project_path = ?
                ORDER BY created_at DESC
                LIMIT 1
            """, (normalized,))
        else:
            cursor.execute("""
                SELECT * FROM curator_reports
                ORDER BY created_at DESC
                LIMIT 1
            """)

        row = cursor.fetchone()
        if not row:
            return None

        return {
            "id": row["id"],
            "project_path": row["project_path"],
            "report_type": row["report_type"],
            "created_at": row["created_at"],
            "summary": row["summary"],
            "findings": json.loads(row["findings"]) if row["findings"] else {},
            "actions_taken": json.loads(row["actions_taken"]) if row["actions_taken"] else {},
            "recommendations": json.loads(row["recommendations"]) if row["recommendations"] else []
        }

    async def get_status(self) -> Dict[str, Any]:
        """Get current curator agent status."""
        cursor = self.db.conn.cursor()

        # Get total memories
        cursor.execute("SELECT COUNT(*) as total FROM memories")
        total_memories = cursor.fetchone()["total"]

        # Get total relationships
        cursor.execute("SELECT COUNT(*) as total FROM memory_relationships")
        total_relationships = cursor.fetchone()["total"]

        # Get orphan count
        cursor.execute("""
            SELECT COUNT(*) as count FROM memories m
            LEFT JOIN memory_relationships mr1 ON m.id = mr1.source_id
            LEFT JOIN memory_relationships mr2 ON m.id = mr2.target_id
            WHERE mr1.id IS NULL AND mr2.id IS NULL
        """)
        orphan_count = cursor.fetchone()["count"]

        # Get latest report
        latest_report = await self.get_latest_report()

        return {
            "active": True,
            "total_memories": total_memories,
            "total_relationships": total_relationships,
            "orphan_count": orphan_count,
            "connection_ratio": round(total_relationships / max(total_memories, 1), 2),
            "last_maintenance": latest_report.get("created_at") if latest_report else None,
            "last_report_summary": latest_report.get("summary") if latest_report else None
        }


# Singleton instance
_curator_instance: Optional[MemoryCurator] = None


def get_curator(db, embeddings) -> MemoryCurator:
    """Get or create the curator singleton."""
    global _curator_instance
    if _curator_instance is None:
        _curator_instance = MemoryCurator(db, embeddings)
    return _curator_instance


async def run_curator_scheduler(
    db,
    embeddings,
    interval_hours: int = 24
):
    """Background scheduler for curator maintenance."""
    curator = get_curator(db, embeddings)

    while True:
        try:
            # Wait for the interval
            await asyncio.sleep(interval_hours * 3600)

            # Run maintenance
            logger.info("Running scheduled curator maintenance...")
            report = await curator.run_maintenance()
            logger.info(f"Curator maintenance complete: {report.get('summary', '')}")

        except asyncio.CancelledError:
            logger.info("Curator scheduler cancelled")
            break
        except Exception as e:
            logger.error(f"Curator scheduler error: {e}")
            # Continue running despite errors
            await asyncio.sleep(300)  # Wait 5 min before retry
