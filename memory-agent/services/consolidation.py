"""Memory Consolidation Service - CLaRa-inspired salient compression.

Clusters similar warm-tier memories and consolidates them into single
compressed memories, preserving the most salient information.

Inspired by CLaRa's approach of compressing documents into fixed-size
memory tokens while maintaining semantic quality through salience scoring.

Process:
1. Find clusters of similar warm-tier memories (cosine sim >= threshold)
2. Score each memory by salience (outcome, importance, confidence, access)
3. Preserve top 2 full, summarize the rest
4. Create consolidated memory with weighted-average embedding
5. Archive originals with reference to consolidated memory
"""
import json
import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from config import config

logger = logging.getLogger(__name__)


class ConsolidationService:
    """Consolidates similar memories to reduce redundancy and improve search.

    Only operates on warm-tier memories to avoid disrupting hot (active) content.
    """

    def __init__(self, db, embeddings=None):
        self.db = db
        self.embeddings = embeddings
        self.similarity_threshold = config.CONSOLIDATION_THRESHOLD
        self.min_group_size = config.CONSOLIDATION_MIN_GROUP
        self.max_group_size = config.CONSOLIDATION_MAX_GROUP
        self.max_per_run = config.CONSOLIDATION_MAX_PER_RUN

    def _calculate_salience(self, memory: dict) -> float:
        """Calculate salience score for a memory.

        Salience = outcome_success_bonus + importance/10 + confidence + access_frequency

        Args:
            memory: Memory dict with outcome, importance, confidence, access_count

        Returns:
            Float salience score (higher = more important to preserve)
        """
        score = 0.0

        # Outcome success bonus
        outcome_status = memory.get('outcome_status', 'pending')
        if outcome_status == 'success':
            score += 3.0
        elif outcome_status == 'partial':
            score += 1.5

        success = memory.get('success')
        if success:
            score += 2.0

        # Importance (normalized 0-1)
        importance = memory.get('importance', 5) or 5
        score += importance / 10.0

        # Confidence
        confidence = memory.get('confidence', 0.5) or 0.5
        score += confidence

        # Access frequency (log scale, capped)
        access_count = memory.get('access_count', 0) or 0
        import math
        score += 0.2 * math.log(1 + min(access_count, 50))

        return round(score, 4)

    async def find_consolidation_candidates(self) -> List[List[dict]]:
        """Find groups of similar warm-tier memories that could be consolidated.

        Uses pairwise cosine similarity on warm-tier memory embeddings.

        Returns:
            List of groups, where each group is a list of similar memory dicts.
        """
        cursor = self.db.conn.cursor()

        # Get warm-tier memories with embeddings
        cursor.execute("""
            SELECT id, type, content, embedding, importance, confidence,
                   access_count, outcome_status, success, created_at,
                   project_path, metadata, tags, outcome
            FROM memories
            WHERE (tier = 'warm' OR (tier IS NULL AND importance < 7))
            AND embedding IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 500
        """)

        rows = cursor.fetchall()
        if len(rows) < self.min_group_size:
            return []

        # Deserialize embeddings
        memories = []
        embeddings_list = []
        for row in rows:
            emb = self.db._deserialize_embedding(row['embedding'])
            if emb:
                memories.append(dict(row))
                embeddings_list.append(emb)

        if len(memories) < self.min_group_size:
            return []

        # Compute pairwise cosine similarity matrix
        emb_matrix = np.array(embeddings_list, dtype=np.float32)
        # Normalize
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        emb_matrix = emb_matrix / norms

        similarity_matrix = emb_matrix @ emb_matrix.T

        # Greedy clustering: find groups above threshold
        used = set()
        groups = []

        for i in range(len(memories)):
            if i in used:
                continue

            group_indices = [i]
            for j in range(i + 1, len(memories)):
                if j in used:
                    continue
                if similarity_matrix[i][j] >= self.similarity_threshold:
                    group_indices.append(j)
                    if len(group_indices) >= self.max_group_size:
                        break

            if len(group_indices) >= self.min_group_size:
                group = [memories[idx] for idx in group_indices]
                groups.append(group)
                used.update(group_indices)

        return groups

    async def consolidate_group(self, group: List[dict]) -> Optional[Dict[str, Any]]:
        """Consolidate a group of similar memories into one.

        Strategy:
        1. Score each by salience
        2. Top 2: preserve full content
        3. Remaining: first 100 chars as summary
        4. Embedding: weighted average by salience
        5. Best metadata from the group

        Args:
            group: List of similar memory dicts

        Returns:
            Dict with consolidated memory info, or None on failure
        """
        if len(group) < self.min_group_size:
            return None

        # Score by salience
        scored = [(mem, self._calculate_salience(mem)) for mem in group]
        scored.sort(key=lambda x: x[1], reverse=True)

        # Build consolidated content
        content_parts = []
        for i, (mem, salience) in enumerate(scored):
            if i < 2:
                # Top 2: full content
                content_parts.append(mem['content'])
            else:
                # Rest: truncated summary
                truncated = mem['content'][:100]
                if len(mem['content']) > 100:
                    truncated += '...'
                content_parts.append(f"[Related] {truncated}")

        consolidated_content = '\n\n---\n\n'.join(content_parts)

        # Weighted average embedding
        embeddings = []
        weights = []
        for mem, salience in scored:
            emb = self.db._deserialize_embedding(mem.get('embedding', ''))
            if emb:
                embeddings.append(emb)
                weights.append(salience)

        consolidated_embedding = None
        if embeddings:
            emb_array = np.array(embeddings, dtype=np.float32)
            weight_array = np.array(weights, dtype=np.float32)
            weight_array = weight_array / weight_array.sum()  # Normalize
            consolidated_embedding = (emb_array * weight_array[:, np.newaxis]).sum(axis=0).tolist()

        # Best metadata from group
        best_mem, best_salience = scored[0]
        best_importance = max(m.get('importance', 5) or 5 for m, _ in scored)
        best_confidence = max(m.get('confidence', 0.5) or 0.5 for m, _ in scored)
        best_outcome = None
        for m, _ in scored:
            if m.get('outcome_status') == 'success':
                best_outcome = 'success'
                break
            elif m.get('outcome_status') == 'partial':
                best_outcome = 'partial'

        source_ids = [m['id'] for m, _ in scored]

        # Merge metadata
        merged_metadata = {
            'consolidated': True,
            'source_ids': source_ids,
            'consolidation_strategy': 'salient_compression',
            'consolidated_at': datetime.now().isoformat(),
            'group_size': len(group),
            'salience_scores': {m['id']: s for m, s in scored}
        }

        # Insert consolidated memory
        cursor = self.db.conn.cursor()
        try:
            embedding_str = self.db._serialize_embedding(consolidated_embedding) if consolidated_embedding else None

            cursor.execute("""
                INSERT INTO memories
                (type, content, embedding, project_path, importance, confidence,
                 metadata, outcome_status, tier, tier_changed_at, tags, outcome)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'warm', ?, ?, ?)
            """, (
                best_mem.get('type', 'chunk'),
                consolidated_content,
                embedding_str,
                best_mem.get('project_path'),
                best_importance,
                best_confidence,
                json.dumps(merged_metadata),
                best_outcome or 'pending',
                datetime.now().isoformat(),
                best_mem.get('tags'),
                best_mem.get('outcome')
            ))

            consolidated_id = cursor.lastrowid

            # Archive originals
            for mem, salience in scored:
                cursor.execute("""
                    INSERT INTO memory_archive
                    (original_id, type, content, embedding, project_path, session_id,
                     importance, access_count, decay_factor, metadata,
                     archive_reason, consolidated_into)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'consolidated', ?)
                """, (
                    mem['id'], mem.get('type'), mem['content'], mem.get('embedding'),
                    mem.get('project_path'), mem.get('session_id'),
                    mem.get('importance'), mem.get('access_count'),
                    mem.get('decay_factor'), mem.get('metadata'),
                    consolidated_id
                ))

                # Delete original from active memories
                cursor.execute("DELETE FROM memories WHERE id = ?", (mem['id'],))

            self.db.conn.commit()

            # Add to FAISS index if available
            if consolidated_embedding and hasattr(self.db, '_memories_index') and self.db._memories_index:
                self.db._memories_index.add(consolidated_id, consolidated_embedding)

            logger.info(
                f"Consolidated {len(group)} memories into memory {consolidated_id} "
                f"(sources: {source_ids})"
            )

            return {
                'consolidated_id': consolidated_id,
                'source_ids': source_ids,
                'group_size': len(group),
                'content_length': len(consolidated_content),
                'best_importance': best_importance,
                'best_confidence': best_confidence
            }

        except Exception as e:
            self.db.conn.rollback()
            logger.error(f"Failed to consolidate group: {e}")
            return None

    async def run_consolidation(self) -> Dict[str, Any]:
        """Run a consolidation pass.

        Finds candidates and consolidates up to max_per_run groups.

        Returns:
            Dict with consolidation statistics
        """
        groups = await self.find_consolidation_candidates()

        results = {
            'candidates_found': len(groups),
            'consolidated': 0,
            'memories_archived': 0,
            'consolidations': [],
            'timestamp': datetime.now().isoformat()
        }

        for group in groups[:self.max_per_run]:
            result = await self.consolidate_group(group)
            if result:
                results['consolidated'] += 1
                results['memories_archived'] += result['group_size']
                results['consolidations'].append(result)

        return results

    async def deconsolidate(self, consolidated_id: int) -> Dict[str, Any]:
        """Restore original memories from a consolidated memory.

        Args:
            consolidated_id: ID of the consolidated memory

        Returns:
            Dict with restoration details
        """
        cursor = self.db.conn.cursor()

        # Find archived originals
        cursor.execute("""
            SELECT * FROM memory_archive
            WHERE consolidated_into = ?
        """, (consolidated_id,))

        archived = cursor.fetchall()
        if not archived:
            return {'success': False, 'error': 'No archived memories found for this consolidation'}

        restored_ids = []
        try:
            for row in archived:
                # Restore to memories table
                cursor.execute("""
                    INSERT INTO memories
                    (type, content, embedding, project_path, session_id,
                     importance, access_count, decay_factor, metadata, tier)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'warm')
                """, (
                    row['type'], row['content'], row['embedding'],
                    row['project_path'], row['session_id'],
                    row['importance'], row['access_count'],
                    row['decay_factor'], row['metadata']
                ))
                restored_ids.append(cursor.lastrowid)

                # Remove from archive
                cursor.execute("DELETE FROM memory_archive WHERE id = ?", (row['id'],))

            # Delete the consolidated memory
            cursor.execute("DELETE FROM memories WHERE id = ?", (consolidated_id,))

            self.db.conn.commit()

            return {
                'success': True,
                'consolidated_id': consolidated_id,
                'restored_count': len(restored_ids),
                'restored_ids': restored_ids
            }

        except Exception as e:
            self.db.conn.rollback()
            logger.error(f"Failed to deconsolidate memory {consolidated_id}: {e}")
            return {'success': False, 'error': str(e)}

    async def get_consolidation_stats(self) -> Dict[str, Any]:
        """Get statistics about consolidation activity."""
        cursor = self.db.conn.cursor()

        # Count consolidated memories
        cursor.execute("""
            SELECT COUNT(*) as count FROM memories
            WHERE metadata LIKE '%"consolidated": true%'
        """)
        consolidated_count = cursor.fetchone()['count']

        # Count archived by consolidation
        cursor.execute("""
            SELECT COUNT(*) as count FROM memory_archive
            WHERE archive_reason = 'consolidated'
        """)
        archived_count = cursor.fetchone()['count']

        # Average group size
        cursor.execute("""
            SELECT AVG(json_extract(metadata, '$.group_size')) as avg_group_size
            FROM memories
            WHERE metadata LIKE '%"consolidated": true%'
        """)
        row = cursor.fetchone()
        avg_group_size = round(row['avg_group_size'] or 0, 1)

        return {
            'consolidated_memories': consolidated_count,
            'archived_originals': archived_count,
            'avg_group_size': avg_group_size,
            'space_savings_estimate': f"{archived_count - consolidated_count} memories removed",
            'timestamp': datetime.now().isoformat()
        }
