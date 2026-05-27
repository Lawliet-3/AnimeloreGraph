"""
Vector embeddings abstraction layer.

Provides a backend-agnostic interface for storing and querying node
embeddings.  Concrete backends (Qdrant, Milvus, in-memory) implement the
``VectorStore`` protocol.

The in-memory backend is used by default and is suitable for development,
testing, and small-scale deployments.  Production deployments should use
the Qdrant or Milvus backends.
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute the cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class VectorStore(ABC):
    """
    Abstract vector store interface.

    All implementations must provide ``upsert`` and ``query`` methods.
    Node IDs must follow the ``universe::name`` convention so that
    universe-scoped filtering can be applied at query time.
    """

    @abstractmethod
    def upsert(self, node_id: str, vector: List[float], metadata: Optional[Dict] = None) -> None:
        """Insert or update a vector entry for *node_id*."""

    @abstractmethod
    def query(
        self,
        query_vector: List[float],
        top_k: int = 10,
        universe_filter: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        Return the *top_k* most similar node IDs and their scores.

        Parameters
        ----------
        query_vector:
            The query embedding.
        top_k:
            Maximum number of results to return.
        universe_filter:
            If provided (e.g. ``'one_piece'``), only nodes from that universe
            are considered, preventing cross-universe contamination in
            semantic search results.
        """

    @abstractmethod
    def delete(self, node_id: str) -> None:
        """Remove the vector entry for *node_id*."""


# ---------------------------------------------------------------------------
# In-memory backend (default)
# ---------------------------------------------------------------------------

class InMemoryVectorStore(VectorStore):
    """
    Pure-Python in-memory vector store using cosine similarity.

    Not recommended for large corpora, but useful for local development
    and tests without requiring an external vector database.
    """

    def __init__(self) -> None:
        # node_id → (vector, metadata)
        self._store: Dict[str, Tuple[List[float], Dict]] = {}

    def upsert(
        self,
        node_id: str,
        vector: List[float],
        metadata: Optional[Dict] = None,
    ) -> None:
        self._store[node_id] = (vector, metadata or {})
        logger.debug("Upserted vector for node %s", node_id)

    def query(
        self,
        query_vector: List[float],
        top_k: int = 10,
        universe_filter: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        scores: List[Tuple[str, float]] = []
        prefix = f"{universe_filter}::" if universe_filter else None

        for node_id, (vec, _meta) in self._store.items():
            if prefix and not node_id.startswith(prefix):
                continue
            sim = _cosine_similarity(query_vector, vec)
            scores.append((node_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def delete(self, node_id: str) -> None:
        self._store.pop(node_id, None)
        logger.debug("Deleted vector for node %s", node_id)

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Qdrant backend (optional dependency)
# ---------------------------------------------------------------------------

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        Distance,
        FieldCondition,
        Filter,
        MatchValue,
        PointStruct,
        VectorParams,
    )

    _QDRANT_AVAILABLE = True
except ImportError:
    _QDRANT_AVAILABLE = False


class QdrantVectorStore(VectorStore):
    """
    Qdrant-backed vector store.

    Requires the ``qdrant-client`` package to be installed.

    Parameters
    ----------
    url:
        Qdrant server URL (e.g. ``'http://localhost:6333'``).
    collection_name:
        Name of the Qdrant collection to use.
    vector_size:
        Dimensionality of the embedding vectors.
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = "animelore",
        vector_size: int = 1536,
    ) -> None:
        if not _QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client is not installed. "
                "Install it with: pip install qdrant-client"
            )
        self._client = QdrantClient(url=url)
        self._collection = collection_name
        self._vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=self._vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info("Created Qdrant collection '%s'", self._collection)

    def upsert(
        self,
        node_id: str,
        vector: List[float],
        metadata: Optional[Dict] = None,
    ) -> None:
        payload = {"node_id": node_id}
        if metadata:
            payload.update(metadata)
        universe = node_id.split("::")[0] if "::" in node_id else "unknown"
        payload["universe"] = universe

        # Use a deterministic integer ID derived from the string
        point_id = abs(hash(node_id)) % (2**31)
        self._client.upsert(
            collection_name=self._collection,
            points=[PointStruct(id=point_id, vector=vector, payload=payload)],
        )
        logger.debug("Upserted Qdrant point for node %s", node_id)

    def query(
        self,
        query_vector: List[float],
        top_k: int = 10,
        universe_filter: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        query_filter = None
        if universe_filter:
            query_filter = Filter(
                must=[
                    FieldCondition(
                        key="universe",
                        match=MatchValue(value=universe_filter),
                    )
                ]
            )
        results = self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [(r.payload["node_id"], r.score) for r in results]

    def delete(self, node_id: str) -> None:
        point_id = abs(hash(node_id)) % (2**31)
        self._client.delete(
            collection_name=self._collection,
            points_selector=[point_id],
        )
        logger.debug("Deleted Qdrant point for node %s", node_id)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_vector_store(backend: str = "memory", **kwargs) -> VectorStore:
    """
    Factory that returns a ``VectorStore`` instance for the chosen backend.

    Parameters
    ----------
    backend:
        One of ``'memory'``, ``'qdrant'``.
    **kwargs:
        Passed to the backend constructor.

    Returns
    -------
    VectorStore
    """
    if backend == "memory":
        return InMemoryVectorStore()
    if backend == "qdrant":
        return QdrantVectorStore(**kwargs)
    raise ValueError(
        f"Unknown vector store backend '{backend}'. "
        "Supported backends: 'memory', 'qdrant'."
    )
