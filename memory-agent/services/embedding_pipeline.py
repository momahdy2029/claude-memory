"""Embedding Pipeline - LRU cache, batch generation, and pre-computation.

CLaRa-inspired pre-processing pipeline for embeddings:
1. LRU cache for query embeddings (common searches return instantly)
2. Batch generation via asyncio.gather for throughput
3. Background pre-computation for memories missing embeddings
"""
import hashlib
import logging
import asyncio
from collections import OrderedDict
from typing import List, Optional, Dict, Any

from config import config

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """LRU cache for embedding queries.

    MD5 hash of text -> embedding vector.
    ~1.5MB footprint for 500 entries at 768 dimensions.
    """

    def __init__(self, max_size: int = None):
        self.max_size = max_size or config.EMBEDDING_CACHE_SIZE
        self._cache: OrderedDict[str, List[float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _key(self, text: str) -> str:
        """Generate cache key from text."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text.

        Args:
            text: Input text

        Returns:
            Cached embedding or None
        """
        key = self._key(text)
        if key in self._cache:
            self._hits += 1
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, text: str, embedding: List[float]):
        """Cache an embedding.

        Args:
            text: Input text
            embedding: Embedding vector
        """
        key = self._key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)  # Remove oldest
        self._cache[key] = embedding

    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': round(self._hits / total, 4) if total > 0 else 0.0,
            'estimated_memory_mb': round(len(self._cache) * 768 * 4 / 1024 / 1024, 2)
        }


class EmbeddingPipeline:
    """Manages embedding generation with caching and batch processing.

    Wraps an EmbeddingService with:
    - LRU query cache
    - Batch generation via asyncio.gather
    - Background pre-computation for missing embeddings
    """

    def __init__(self, embedding_service, db=None):
        self.embedding_service = embedding_service
        self.db = db
        self.cache = EmbeddingCache()
        self.batch_size = config.EMBEDDING_BATCH_SIZE
        self._precompute_running = False

    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding with LRU cache.

        Args:
            text: Text to embed

        Returns:
            Embedding vector or None if service unavailable
        """
        # Check cache first
        cached = self.cache.get(text)
        if cached is not None:
            return cached

        # Generate new embedding
        embedding = await self.embedding_service.generate_embedding(text)

        # Cache if successful
        if embedding is not None:
            self.cache.put(text, embedding)

        return embedding

    async def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts using concurrent batches.

        Processes in batches of self.batch_size, each batch runs concurrently
        via asyncio.gather against Ollama.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings (or None for failed ones)
        """
        results = [None] * len(texts)

        # Check cache for existing embeddings
        uncached_indices = []
        for i, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)

        if not uncached_indices:
            return results

        # Process uncached texts in concurrent batches
        for batch_start in range(0, len(uncached_indices), self.batch_size):
            batch_indices = uncached_indices[batch_start:batch_start + self.batch_size]
            batch_texts = [texts[i] for i in batch_indices]

            # Generate all in parallel
            batch_results = await asyncio.gather(
                *[self.embedding_service.generate_embedding(text) for text in batch_texts],
                return_exceptions=True
            )

            for idx, emb_result in zip(batch_indices, batch_results):
                if isinstance(emb_result, Exception):
                    logger.debug(f"Batch embedding failed for index {idx}: {emb_result}")
                    continue
                if emb_result is not None:
                    results[idx] = emb_result
                    self.cache.put(texts[idx], emb_result)

        return results

    async def precompute_missing_embeddings(self) -> Dict[str, Any]:
        """Background task: find memories with NULL embeddings and generate them.

        Only runs when Ollama is healthy (not degraded).

        Returns:
            Dict with precomputation stats
        """
        if self._precompute_running:
            return {'skipped': True, 'reason': 'already_running'}

        if self.embedding_service.is_degraded():
            return {'skipped': True, 'reason': 'ollama_degraded'}

        if not self.db:
            return {'skipped': True, 'reason': 'no_db'}

        self._precompute_running = True
        try:
            cursor = self.db.conn.cursor()

            # Find memories with missing embeddings
            cursor.execute("""
                SELECT id, content FROM memories
                WHERE embedding IS NULL
                ORDER BY importance DESC, created_at DESC
                LIMIT ?
            """, (self.batch_size * 5,))  # Process up to 50 at a time

            rows = cursor.fetchall()
            if not rows:
                return {'generated': 0, 'message': 'all_embeddings_present'}

            # Generate embeddings in batch
            texts = [row['content'] for row in rows]
            embeddings = await self.generate_embeddings_batch(texts)

            # Update database
            updated = 0
            for row, emb in zip(rows, embeddings):
                if emb is not None:
                    emb_str = self.db._serialize_embedding(emb)
                    cursor.execute(
                        "UPDATE memories SET embedding = ? WHERE id = ?",
                        (emb_str, row['id'])
                    )

                    # Add to FAISS index if available
                    if hasattr(self.db, '_memories_index') and self.db._memories_index:
                        self.db._memories_index.add(row['id'], emb)

                    updated += 1

            if updated:
                self.db.conn.commit()

            return {
                'found_missing': len(rows),
                'generated': updated,
                'failed': len(rows) - updated
            }

        except Exception as e:
            logger.error(f"Precompute failed: {e}")
            return {'error': str(e)}

        finally:
            self._precompute_running = False

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            'cache': self.cache.get_stats(),
            'batch_size': self.batch_size,
            'precompute_running': self._precompute_running,
            'service_degraded': self.embedding_service.is_degraded()
        }


# Global pipeline instance
_pipeline: Optional[EmbeddingPipeline] = None


def get_embedding_pipeline(embedding_service=None, db=None) -> EmbeddingPipeline:
    """Get or create the global embedding pipeline."""
    global _pipeline
    if _pipeline is None and embedding_service:
        _pipeline = EmbeddingPipeline(embedding_service, db)
    return _pipeline
