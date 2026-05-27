"""
Query engine for multi-hop graph traversal and semantic search.

Combines graph-structural queries (BFS/DFS path-finding in the
``GraphStore``) with vector-based semantic lookup (``VectorStore``) to
answer complex, multi-hop questions across anime universe knowledge graphs.

Query results are always universe-scoped to prevent cross-universe
data contamination.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .embeddings import VectorStore
from .graph_store import GraphStore
from .models import NodeType, RelationType, Universe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PathResult:
    """A single multi-hop path between two nodes."""

    source_id: str
    target_id: str
    path: List[str]
    hops: int = field(init=False)
    edges: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.hops = len(self.path) - 1


@dataclass
class SemanticResult:
    """A single semantic search hit."""

    node_id: str
    score: float
    node_data: Optional[Dict[str, Any]] = None


@dataclass
class AggregateResult:
    """Universe-level aggregate statistics."""

    universe: str
    total_nodes: int
    total_edges: int
    node_type_counts: Dict[str, int]
    relation_type_counts: Dict[str, int]


# ---------------------------------------------------------------------------
# Query engine
# ---------------------------------------------------------------------------

class QueryEngine:
    """
    High-level query interface combining graph traversal and semantic search.

    Parameters
    ----------
    graph_store:
        The ``GraphStore`` instance containing the knowledge graph.
    vector_store:
        An optional ``VectorStore`` for embedding-based semantic lookup.
        When ``None``, semantic search methods raise ``RuntimeError``.
    embed_fn:
        A callable ``(text: str) -> List[float]`` that produces embeddings.
        Required when a *vector_store* is provided.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: Optional[VectorStore] = None,
        embed_fn: Optional[Any] = None,
    ) -> None:
        self._graph = graph_store
        self._vector = vector_store
        self._embed = embed_fn

    # ------------------------------------------------------------------
    # Multi-hop traversal queries
    # ------------------------------------------------------------------

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 4,
        relation_filter: Optional[List[RelationType]] = None,
    ) -> List[PathResult]:
        """
        Find all simple paths between *source_id* and *target_id*.

        Cross-universe paths are never returned because the graph does not
        store cross-universe edges.

        Parameters
        ----------
        source_id:
            Starting node (must include universe prefix).
        target_id:
            Destination node (must include universe prefix).
        max_hops:
            Maximum path length in edges.
        relation_filter:
            If provided, only traverse edges of these types.

        Returns
        -------
        List[PathResult]
            Sorted by path length (shortest first).
        """
        raw_paths = self._graph.multi_hop_paths(
            source_id, target_id, max_hops=max_hops, relation_filter=relation_filter
        )
        results: List[PathResult] = []
        for path in raw_paths:
            edges = self._path_edges(path)
            results.append(
                PathResult(source_id=source_id, target_id=target_id, path=path, edges=edges)
            )
        results.sort(key=lambda r: r.hops)
        return results

    def _path_edges(self, path: List[str]) -> List[Dict[str, Any]]:
        """Return the edge attribute dicts for each step in *path*."""
        edges = []
        for i in range(len(path) - 1):
            edge_list = self._graph.get_edges(source_id=path[i], target_id=path[i + 1])
            edges.extend(edge_list)
        return edges

    def get_character_connections(
        self,
        character_id: str,
        depth: int = 2,
    ) -> Dict[str, Any]:
        """
        Return a subgraph centred on *character_id* up to *depth* hops.

        The universe is inferred from the character's ID prefix, ensuring
        the returned subgraph never spans multiple universes.
        """
        universe_str = character_id.split("::")[0]
        try:
            universe = Universe(universe_str)
        except ValueError:
            raise ValueError(
                f"Cannot infer universe from node ID '{character_id}'."
            )
        sub = self._graph.bfs_subgraph(
            character_id, max_depth=depth, universe=universe
        )
        return sub.to_dict()

    def find_shared_events(
        self,
        char_a: str,
        char_b: str,
    ) -> List[str]:
        """
        Return event node IDs that both *char_a* and *char_b* participated in.

        Only events within the same universe are considered.
        """
        a_events = {
            nid
            for nid in self._graph.neighbors(
                char_a, relation=RelationType.participated_in, direction="out"
            )
        }
        b_events = {
            nid
            for nid in self._graph.neighbors(
                char_b, relation=RelationType.participated_in, direction="out"
            )
        }
        return sorted(a_events & b_events)

    def find_characters_in_faction(
        self,
        faction_id: str,
    ) -> List[str]:
        """
        Return all character IDs that are members of *faction_id*.
        """
        return self._graph.neighbors(
            faction_id,
            relation=RelationType.member_of,
            direction="in",
        )

    def find_abilities_of_character(
        self,
        character_id: str,
    ) -> List[str]:
        """
        Return the ability node IDs wielded by *character_id*.
        """
        return self._graph.neighbors(
            character_id,
            relation=RelationType.wields,
            direction="out",
        )

    # ------------------------------------------------------------------
    # Semantic search
    # ------------------------------------------------------------------

    def semantic_search(
        self,
        query_text: str,
        top_k: int = 10,
        universe: Optional[Universe] = None,
    ) -> List[SemanticResult]:
        """
        Find nodes semantically similar to *query_text*.

        If *universe* is specified, results are constrained to that universe
        (cross-universe contamination prevention).

        Raises
        ------
        RuntimeError
            If no ``VectorStore`` or embedding function was provided at
            construction time.
        """
        if self._vector is None or self._embed is None:
            raise RuntimeError(
                "semantic_search requires a VectorStore and embed_fn. "
                "Provide these when constructing QueryEngine."
            )
        query_vector = self._embed(query_text)
        universe_filter = universe.value if universe else None
        hits = self._vector.query(
            query_vector, top_k=top_k, universe_filter=universe_filter
        )
        results = []
        for node_id, score in hits:
            node_data = self._graph.get_node(node_id)
            results.append(SemanticResult(node_id=node_id, score=score, node_data=node_data))
        return results

    def index_node(self, node_id: str, text: str) -> None:
        """
        Embed *text* and upsert the resulting vector for *node_id*.

        Used to populate the vector store from node descriptions so that
        semantic search returns meaningful results.
        """
        if self._vector is None or self._embed is None:
            raise RuntimeError(
                "index_node requires a VectorStore and embed_fn."
            )
        vector = self._embed(text)
        metadata = self._graph.get_node(node_id) or {}
        self._vector.upsert(node_id, vector, metadata=metadata)

    # ------------------------------------------------------------------
    # Aggregate queries
    # ------------------------------------------------------------------

    def aggregate_stats(
        self,
        universe: Optional[Universe] = None,
    ) -> List[AggregateResult]:
        """
        Return aggregate counts per universe (or just for *universe* if given).

        Parameters
        ----------
        universe:
            If provided, only statistics for that universe are returned.

        Returns
        -------
        List[AggregateResult]
        """
        universes = [universe] if universe else list(Universe)
        results = []
        for u in universes:
            nodes = self._graph.nodes_by_universe(u)
            edges = self._graph.get_edges()
            prefix = f"{u.value}::"

            node_type_counts: Dict[str, int] = {}
            for node in nodes:
                nt = node.get("node_type", "unknown")
                node_type_counts[nt] = node_type_counts.get(nt, 0) + 1

            rel_type_counts: Dict[str, int] = {}
            for edge in edges:
                if not (
                    edge.get("source_id", "").startswith(prefix)
                    and edge.get("target_id", "").startswith(prefix)
                ):
                    continue
                rt = edge.get("relation", "unknown")
                rel_type_counts[rt] = rel_type_counts.get(rt, 0) + 1

            results.append(
                AggregateResult(
                    universe=u.value,
                    total_nodes=len(nodes),
                    total_edges=sum(rel_type_counts.values()),
                    node_type_counts=node_type_counts,
                    relation_type_counts=rel_type_counts,
                )
            )
        return results

    def find_highly_connected_nodes(
        self,
        universe: Optional[Universe] = None,
        top_k: int = 10,
    ) -> List[Tuple[str, int]]:
        """
        Return the *top_k* nodes with the highest degree (in + out edges).

        Optionally scoped to a single *universe*.
        """
        prefix = f"{universe.value}::" if universe else None
        degree_list: List[Tuple[str, int]] = []
        for nid in self._graph._graph.nodes():
            if prefix and not nid.startswith(prefix):
                continue
            degree = (
                self._graph._graph.in_degree(nid)
                + self._graph._graph.out_degree(nid)
            )
            degree_list.append((nid, degree))
        degree_list.sort(key=lambda x: x[1], reverse=True)
        return degree_list[:top_k]
