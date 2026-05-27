"""
AnimeloreGraph pipeline orchestrator.

Provides a high-level ``AnimeloreGraphPipeline`` class that wires together
the extraction, storage, embedding, and query layers into a cohesive
end-to-end pipeline for multi-universe anime lore knowledge graph
construction and querying.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .embeddings import InMemoryVectorStore, VectorStore, create_vector_store
from .extractor import KnowledgeExtractor
from .graph_store import GraphStore
from .models import Universe
from .query_engine import (
    AggregateResult,
    PathResult,
    QueryEngine,
    SemanticResult,
)

logger = logging.getLogger(__name__)


class AnimeloreGraphPipeline:
    """
    End-to-end GraphRAG pipeline for multi-universe anime lore.

    Responsibilities
    ----------------
    1. **Ingestion**: Accept raw text passages tagged with a universe,
       run LLM extraction, convert to graph nodes/edges, and store them.
    2. **Indexing**: Optionally embed node descriptions into a vector store
       for semantic search.
    3. **Querying**: Expose a unified query surface that delegates to
       ``QueryEngine`` for multi-hop traversal and semantic search.
    4. **Persistence**: Save and load the knowledge graph to/from disk.

    Parameters
    ----------
    graph_store:
        Pre-configured ``GraphStore`` instance (defaults to a new empty one).
    vector_store:
        Pre-configured ``VectorStore`` instance (defaults to in-memory).
    extractor:
        Pre-configured ``KnowledgeExtractor`` (defaults to one using the
        ``gpt-4o-mini`` model).
    embed_fn:
        A callable ``(text: str) -> List[float]`` used to embed node
        descriptions for semantic search.  When ``None``, semantic search
        is disabled.
    openai_api_key:
        OpenAI API key forwarded to ``KnowledgeExtractor`` when no explicit
        *extractor* is provided.
    """

    def __init__(
        self,
        graph_store: Optional[GraphStore] = None,
        vector_store: Optional[VectorStore] = None,
        extractor: Optional[KnowledgeExtractor] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        openai_api_key: Optional[str] = None,
    ) -> None:
        self._graph = graph_store or GraphStore()
        self._vectors: VectorStore = vector_store or InMemoryVectorStore()
        self._extractor: KnowledgeExtractor = extractor or KnowledgeExtractor(
            api_key=openai_api_key
        )
        self._embed_fn = embed_fn
        self._query_engine = QueryEngine(
            graph_store=self._graph,
            vector_store=self._vectors,
            embed_fn=embed_fn,
        )

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        text: str,
        universe: Universe,
        auto_index: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract knowledge from *text*, insert nodes and edges into the graph,
        and optionally embed node descriptions into the vector store.

        Parameters
        ----------
        text:
            Raw passage from the target fictional universe.
        universe:
            Which universe the text belongs to (prevents cross-contamination).
        auto_index:
            If ``True`` and an embedding function is configured, automatically
            index each new node's description in the vector store.

        Returns
        -------
        dict
            Summary with keys ``'nodes_added'`` and ``'edges_added'``.
        """
        result = self._extractor.extract(text, universe)
        nodes, relationships = self._extractor.extraction_to_graph_objects(result)

        nodes_added = 0
        for node in nodes:
            existed = self._graph.has_node(node.id)
            self._graph.add_node(node)
            if not existed:
                nodes_added += 1
                if auto_index and self._embed_fn is not None:
                    desc = node.description or node.name
                    self._query_engine.index_node(node.id, desc)

        edges_added = 0
        for rel in relationships:
            try:
                self._graph.add_edge(rel)
                edges_added += 1
            except (KeyError, ValueError) as exc:
                logger.warning("Skipped edge: %s", exc)

        logger.info(
            "[%s] Ingested passage: +%d nodes, +%d edges",
            universe.value,
            nodes_added,
            edges_added,
        )
        return {"nodes_added": nodes_added, "edges_added": edges_added}

    def ingest_batch(
        self,
        passages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Ingest a batch of ``{'text': ..., 'universe': Universe}`` dicts.

        Parameters
        ----------
        passages:
            Each dict must have ``'text'`` (str) and ``'universe'``
            (``Universe`` or its string value).

        Returns
        -------
        List of per-passage ingestion summaries.
        """
        summaries = []
        for item in passages:
            universe = item["universe"]
            if isinstance(universe, str):
                universe = Universe(universe)
            summary = self.ingest(text=item["text"], universe=universe)
            summaries.append(summary)
        return summaries

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query_paths(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 4,
    ) -> List[PathResult]:
        """Find multi-hop paths between two nodes."""
        return self._query_engine.find_paths(source_id, target_id, max_hops=max_hops)

    def query_semantic(
        self,
        query_text: str,
        top_k: int = 10,
        universe: Optional[Universe] = None,
    ) -> List[SemanticResult]:
        """Semantic search across node descriptions."""
        return self._query_engine.semantic_search(query_text, top_k=top_k, universe=universe)

    def query_aggregate(
        self,
        universe: Optional[Universe] = None,
    ) -> List[AggregateResult]:
        """Return aggregate node/edge counts per universe."""
        return self._query_engine.aggregate_stats(universe=universe)

    def character_connections(
        self,
        character_id: str,
        depth: int = 2,
    ) -> Dict[str, Any]:
        """Return the BFS neighbourhood subgraph for a character."""
        return self._query_engine.get_character_connections(character_id, depth=depth)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Persist the knowledge graph to disk at *path*."""
        self._graph.save(path)

    @classmethod
    def load(
        cls,
        path: str,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
        openai_api_key: Optional[str] = None,
    ) -> "AnimeloreGraphPipeline":
        """
        Load a previously saved pipeline from *path*.

        The ``GraphStore`` is restored from disk; the ``VectorStore`` starts
        empty (re-index as needed).
        """
        graph = GraphStore.load(path)
        pipeline = cls(
            graph_store=graph,
            embed_fn=embed_fn,
            openai_api_key=openai_api_key,
        )
        logger.info("Pipeline loaded from %s", path)
        return pipeline

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def graph(self) -> GraphStore:
        """Access to the underlying ``GraphStore``."""
        return self._graph

    @property
    def query_engine(self) -> QueryEngine:
        """Access to the underlying ``QueryEngine``."""
        return self._query_engine

    def stats(self) -> Dict[str, Any]:
        """Return high-level stats about the pipeline's knowledge graph."""
        return {
            "total_nodes": self._graph.node_count,
            "total_edges": self._graph.edge_count,
            "universe_breakdown": self._graph.universe_stats(),
        }
