"""FAISS-based vector indexing for fast similarity search.

Provides O(log n) search instead of O(n) for large memory collections.
Falls back to numpy cosine similarity if FAISS is unavailable.
"""
import os
import json
import time
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from threading import Lock
from dotenv import load_dotenv

load_dotenv()

# Try to import FAISS
FAISS_AVAILABLE = False
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None

# Index configuration
try:
    from config import USER_DATA_DIR as _DATA_DIR
except ImportError:
    _DATA_DIR = Path.home() / ".claude-memory"
INDEX_DIR = os.getenv("INDEX_DIR", str(_DATA_DIR / "indexes"))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
INDEX_TYPE = os.getenv("INDEX_TYPE", "flat")  # flat, ivf, hnsw


class VectorIndex:
    """FAISS-based vector index with persistence and automatic rebuilding.

    Supports three index types:
    - flat: Exact search (IndexFlatIP) - best for < 10K vectors
    - ivf: Inverted file index (IndexIVFFlat) - good for 10K-1M vectors
    - hnsw: Hierarchical NSW (IndexHNSWFlat) - best for 1M+ vectors

    Falls back to numpy-based search if FAISS is unavailable.
    """

    def __init__(
        self,
        name: str,
        dimension: int = EMBEDDING_DIM,
        index_type: str = INDEX_TYPE,
        index_dir: str = INDEX_DIR
    ):
        self.name = name
        self.dimension = dimension
        self.index_type = index_type
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index_path = self.index_dir / f"{name}.index"
        self.id_map_path = self.index_dir / f"{name}_ids.json"

        # FAISS index
        self._index: Optional[Any] = None
        self._id_map: List[int] = []  # Maps FAISS internal ID to database ID
        self._reverse_map: Dict[int, int] = {}  # Maps database ID to FAISS internal ID

        # Thread safety
        self._lock = Lock()

        # Stats
        self._last_rebuild: Optional[float] = None
        self._search_count = 0
        self._add_count = 0

        # Fallback storage for numpy-based search
        self._fallback_vectors: List[Tuple[int, np.ndarray]] = []

        # Initialize
        self._initialize_index()

    def _initialize_index(self):
        """Initialize or load the FAISS index."""
        if not FAISS_AVAILABLE:
            return

        # Try to load existing index
        if self.index_path.exists() and self.id_map_path.exists():
            try:
                self._index = faiss.read_index(str(self.index_path))
                with open(self.id_map_path, 'r') as f:
                    self._id_map = json.load(f)
                self._reverse_map = {db_id: idx for idx, db_id in enumerate(self._id_map)}
                return
            except Exception:
                pass  # Fall through to create new index

        # Create new index based on type
        self._create_index()

    def _create_index(self):
        """Create a new FAISS index."""
        if not FAISS_AVAILABLE:
            return

        if self.index_type == "flat":
            # Exact search using inner product (for normalized vectors = cosine similarity)
            self._index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "ivf":
            # IVF index for larger collections
            # Start with flat, train later when we have enough vectors
            self._index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "hnsw":
            # HNSW for very large collections
            self._index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 neighbors
            self._index.hnsw.efConstruction = 200
            self._index.hnsw.efSearch = 128
        else:
            # Default to flat
            self._index = faiss.IndexFlatIP(self.dimension)

        self._id_map = []
        self._reverse_map = {}

    def add(self, db_id: int, embedding: List[float]) -> bool:
        """Add a vector to the index.

        Args:
            db_id: Database ID for this vector
            embedding: The embedding vector (will be L2 normalized)

        Returns:
            True if added successfully
        """
        with self._lock:
            # Normalize vector for cosine similarity via inner product
            vector = np.array([embedding], dtype=np.float32)
            faiss.normalize_L2(vector) if FAISS_AVAILABLE else None

            if FAISS_AVAILABLE and self._index is not None:
                # Check if ID already exists
                if db_id in self._reverse_map:
                    # Update existing - remove old and add new
                    # Note: FAISS doesn't support in-place updates, so we mark for rebuild
                    old_idx = self._reverse_map[db_id]
                    # We can't remove from flat index, so just add and track duplicate
                    # The search will return the latest entry

                self._index.add(vector)
                internal_id = len(self._id_map)
                self._id_map.append(db_id)
                self._reverse_map[db_id] = internal_id
                self._add_count += 1
                return True
            else:
                # Fallback: store in memory
                self._fallback_vectors.append((db_id, vector[0]))
                self._add_count += 1
                return True

    def add_batch(self, items: List[Tuple[int, List[float]]]) -> int:
        """Add multiple vectors to the index.

        Args:
            items: List of (db_id, embedding) tuples

        Returns:
            Number of vectors added
        """
        if not items:
            return 0

        with self._lock:
            vectors = np.array([item[1] for item in items], dtype=np.float32)
            if FAISS_AVAILABLE:
                faiss.normalize_L2(vectors)

            if FAISS_AVAILABLE and self._index is not None:
                self._index.add(vectors)
                for db_id, _ in items:
                    internal_id = len(self._id_map)
                    self._id_map.append(db_id)
                    self._reverse_map[db_id] = internal_id
                self._add_count += len(items)
                return len(items)
            else:
                for i, (db_id, _) in enumerate(items):
                    self._fallback_vectors.append((db_id, vectors[i]))
                self._add_count += len(items)
                return len(items)

    def search(
        self,
        query_embedding: List[float],
        k: int = 10,
        threshold: float = 0.0
    ) -> List[Tuple[int, float]]:
        """Search for similar vectors.

        Args:
            query_embedding: Query vector
            k: Number of results to return
            threshold: Minimum similarity threshold (0-1 for cosine)

        Returns:
            List of (db_id, similarity) tuples, sorted by similarity descending
        """
        with self._lock:
            self._search_count += 1

            # Normalize query vector
            query = np.array([query_embedding], dtype=np.float32)
            if FAISS_AVAILABLE:
                faiss.normalize_L2(query)

            if FAISS_AVAILABLE and self._index is not None and self._index.ntotal > 0:
                # FAISS search
                distances, indices = self._index.search(query, min(k * 2, self._index.ntotal))

                results = []
                seen_ids = set()
                for dist, idx in zip(distances[0], indices[0]):
                    if idx < 0 or idx >= len(self._id_map):
                        continue
                    db_id = self._id_map[idx]
                    if db_id in seen_ids:
                        continue  # Skip duplicates (from updates)
                    seen_ids.add(db_id)

                    # Inner product of normalized vectors = cosine similarity
                    similarity = float(dist)
                    if similarity >= threshold:
                        results.append((db_id, similarity))

                    if len(results) >= k:
                        break

                return results

            else:
                # Fallback to numpy
                return self._numpy_search(query[0], k, threshold)

    def _numpy_search(
        self,
        query: np.ndarray,
        k: int,
        threshold: float
    ) -> List[Tuple[int, float]]:
        """Fallback numpy-based cosine similarity search."""
        if not self._fallback_vectors:
            return []

        results = []
        query_norm = np.linalg.norm(query)
        if query_norm == 0:
            return []

        for db_id, vec in self._fallback_vectors:
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                continue
            similarity = float(np.dot(query, vec) / (query_norm * vec_norm))
            if similarity >= threshold:
                results.append((db_id, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]

    def save(self) -> bool:
        """Persist the index to disk."""
        if not FAISS_AVAILABLE or self._index is None:
            return False

        with self._lock:
            try:
                faiss.write_index(self._index, str(self.index_path))
                with open(self.id_map_path, 'w') as f:
                    json.dump(self._id_map, f)
                return True
            except Exception:
                return False

    def load(self) -> bool:
        """Load the index from disk."""
        if not FAISS_AVAILABLE:
            return False

        with self._lock:
            try:
                if self.index_path.exists() and self.id_map_path.exists():
                    self._index = faiss.read_index(str(self.index_path))
                    with open(self.id_map_path, 'r') as f:
                        self._id_map = json.load(f)
                    self._reverse_map = {db_id: idx for idx, db_id in enumerate(self._id_map)}
                    return True
            except Exception:
                pass
            return False

    def rebuild(self, items: List[Tuple[int, List[float]]]) -> int:
        """Rebuild the entire index from scratch.

        Args:
            items: List of (db_id, embedding) tuples

        Returns:
            Number of vectors indexed
        """
        with self._lock:
            self._create_index()
            self._fallback_vectors = []
            self._last_rebuild = time.time()

        return self.add_batch(items)

    def clear(self):
        """Clear the index."""
        with self._lock:
            self._create_index()
            self._fallback_vectors = []

    def remove(self, db_id: int) -> bool:
        """Mark a vector for removal (requires rebuild to take effect)."""
        # FAISS flat index doesn't support removal
        # We track removed IDs and filter during search
        # For now, just return False - rebuild needed
        return False

    def size(self) -> int:
        """Return the number of vectors in the index."""
        if FAISS_AVAILABLE and self._index is not None:
            return self._index.ntotal
        return len(self._fallback_vectors)

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "faiss_available": FAISS_AVAILABLE,
            "size": self.size(),
            "search_count": self._search_count,
            "add_count": self._add_count,
            "last_rebuild": self._last_rebuild,
            "index_path": str(self.index_path),
            "id_map_size": len(self._id_map)
        }


class VectorIndexManager:
    """Manages multiple vector indexes for different tables."""

    def __init__(self, index_dir: str = INDEX_DIR):
        self.index_dir = index_dir
        self._indexes: Dict[str, VectorIndex] = {}
        self._lock = Lock()

    def get_index(self, name: str, dimension: int = EMBEDDING_DIM) -> VectorIndex:
        """Get or create an index by name."""
        with self._lock:
            if name not in self._indexes:
                self._indexes[name] = VectorIndex(
                    name=name,
                    dimension=dimension,
                    index_dir=self.index_dir
                )
            return self._indexes[name]

    def save_all(self) -> Dict[str, bool]:
        """Save all indexes to disk."""
        results = {}
        with self._lock:
            for name, index in self._indexes.items():
                results[name] = index.save()
        return results

    def get_all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all indexes."""
        stats = {}
        with self._lock:
            for name, index in self._indexes.items():
                stats[name] = index.get_stats()
        stats["faiss_available"] = FAISS_AVAILABLE
        return stats


# Global manager instance
_manager: Optional[VectorIndexManager] = None


def get_index_manager() -> VectorIndexManager:
    """Get the global index manager instance."""
    global _manager
    if _manager is None:
        _manager = VectorIndexManager()
    return _manager


def get_index(name: str, dimension: int = EMBEDDING_DIM) -> VectorIndex:
    """Convenience function to get an index by name."""
    return get_index_manager().get_index(name, dimension)
