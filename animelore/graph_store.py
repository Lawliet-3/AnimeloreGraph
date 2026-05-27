"""
Graph storage layer for the AnimeloreGraph system.

Wraps a ``networkx.MultiDiGraph`` and provides:
- Typed node / edge insertion with namespace-protected IDs.
- Universe-scoped subgraph extraction for cross-contamination prevention.
- Multi-hop BFS/DFS traversal helpers.
- JSON serialisation / deserialisation (no external DB required).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import networkx as nx

from .models import (
    AbilityNode,
    BaseNode,
    CharacterNode,
    EventNode,
    FactionNode,
    LocationNode,
    NodeType,
    Relationship,
    RelationType,
    Universe,
    _validate_node_id,
)

logger = logging.getLogger(__name__)

# Map NodeType enum → concrete model class
_NODE_TYPE_MAP: Dict[NodeType, type] = {
    NodeType.character: CharacterNode,
    NodeType.faction: FactionNode,
    NodeType.ability: AbilityNode,
    NodeType.location: LocationNode,
    NodeType.event: EventNode,
}


class GraphStore:
    """
    In-memory ``networkx.MultiDiGraph``-backed knowledge graph store.

    All node IDs follow the ``universe::name`` convention enforced by the
    ``BaseNode`` validator.  Cross-universe edges are rejected at the
    ``Relationship`` model layer before they reach the graph.
    """

    def __init__(self) -> None:
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_node(self, node: BaseNode) -> None:
        """
        Insert or update a node.

        If a node with the same ``id`` already exists its attributes are
        merged (existing keys are preserved; new keys are added).
        """
        data = node.model_dump()
        if self._graph.has_node(node.id):
            existing = self._graph.nodes[node.id]
            for k, v in data.items():
                if k not in existing:
                    existing[k] = v
            logger.debug("Merged node %s", node.id)
        else:
            self._graph.add_node(node.id, **data)
            logger.debug("Added node %s", node.id)

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Return the attribute dict for a node, or ``None`` if absent."""
        _validate_node_id(node_id)
        if self._graph.has_node(node_id):
            return dict(self._graph.nodes[node_id])
        return None

    def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    def nodes_by_universe(self, universe: Universe) -> List[Dict[str, Any]]:
        """Return all nodes belonging to *universe*."""
        prefix = f"{universe.value}::"
        return [
            dict(data)
            for nid, data in self._graph.nodes(data=True)
            if nid.startswith(prefix)
        ]

    def nodes_by_type(
        self,
        node_type: NodeType,
        universe: Optional[Universe] = None,
    ) -> List[Dict[str, Any]]:
        """Return nodes filtered by type (and optionally by universe)."""
        results = []
        prefix = f"{universe.value}::" if universe else None
        for nid, data in self._graph.nodes(data=True):
            if data.get("node_type") != node_type.value:
                continue
            if prefix and not nid.startswith(prefix):
                continue
            results.append(dict(data))
        return results

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def add_edge(self, relationship: Relationship) -> None:
        """
        Insert a directed edge.  The source and target nodes must already
        exist in the graph.
        """
        if not self._graph.has_node(relationship.source_id):
            raise KeyError(
                f"Source node '{relationship.source_id}' not found in graph. "
                "Add the node before adding an edge."
            )
        if not self._graph.has_node(relationship.target_id):
            raise KeyError(
                f"Target node '{relationship.target_id}' not found in graph. "
                "Add the node before adding an edge."
            )
        edge_data = relationship.model_dump()
        self._graph.add_edge(
            relationship.source_id,
            relationship.target_id,
            key=relationship.relation.value,
            **edge_data,
        )
        logger.debug(
            "Added edge %s -[%s]-> %s",
            relationship.source_id,
            relationship.relation.value,
            relationship.target_id,
        )

    def get_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        relation: Optional[RelationType] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return edges matching the (optional) filter criteria.

        At least one of *source_id*, *target_id*, or *relation* should be
        provided for efficiency; passing none returns all edges.
        """
        results = []
        for u, v, key, data in self._graph.edges(data=True, keys=True):
            if source_id is not None and u != source_id:
                continue
            if target_id is not None and v != target_id:
                continue
            if relation is not None and key != relation.value:
                continue
            results.append({"source_id": u, "target_id": v, "relation": key, **data})
        return results

    # ------------------------------------------------------------------
    # Multi-hop traversal
    # ------------------------------------------------------------------

    def neighbors(
        self,
        node_id: str,
        relation: Optional[RelationType] = None,
        direction: str = "out",
    ) -> List[str]:
        """
        Return direct neighbours of *node_id*.

        Parameters
        ----------
        node_id:
            Starting node.
        relation:
            If provided, only traverse edges of this type.
        direction:
            ``'out'`` (successors), ``'in'`` (predecessors), or
            ``'both'``.
        """
        _validate_node_id(node_id)
        if not self._graph.has_node(node_id):
            return []

        def _filtered(edges: Iterable[Tuple[str, str, str]]) -> List[str]:
            result = []
            for u, v, key in edges:
                if relation is None or key == relation.value:
                    result.append(v if u == node_id else u)
            return result

        if direction == "out":
            raw = list(self._graph.out_edges(node_id, keys=True))
            return _filtered(raw)
        if direction == "in":
            raw = list(self._graph.in_edges(node_id, keys=True))
            return _filtered(raw)
        # both
        out = _filtered(list(self._graph.out_edges(node_id, keys=True)))
        inc = _filtered(list(self._graph.in_edges(node_id, keys=True)))
        seen: Set[str] = set()
        combined = []
        for nid in out + inc:
            if nid not in seen:
                seen.add(nid)
                combined.append(nid)
        return combined

    def multi_hop_paths(
        self,
        source_id: str,
        target_id: str,
        max_hops: int = 4,
        relation_filter: Optional[List[RelationType]] = None,
    ) -> List[List[str]]:
        """
        Return all simple paths from *source_id* to *target_id* up to
        *max_hops* edges using BFS.

        If *relation_filter* is given, only edges whose type is in the list
        are traversed.  Cross-universe paths are inherently impossible
        because cross-universe edges are never stored.
        """
        _validate_node_id(source_id)
        _validate_node_id(target_id)

        if not self._graph.has_node(source_id) or not self._graph.has_node(target_id):
            return []

        if relation_filter is not None:
            allowed_keys = {r.value for r in relation_filter}
            view = nx.subgraph_view(
                self._graph,
                filter_edge=lambda u, v, k: k in allowed_keys,
            )
        else:
            view = self._graph

        try:
            paths = list(
                nx.all_simple_paths(view, source_id, target_id, cutoff=max_hops)
            )
        except (nx.NodeNotFound, nx.NetworkXError):
            paths = []
        return paths

    def bfs_subgraph(
        self,
        root_id: str,
        max_depth: int = 2,
        universe: Optional[Universe] = None,
    ) -> "GraphStore":
        """
        Return a new ``GraphStore`` containing the BFS neighbourhood of
        *root_id* up to *max_depth* hops.

        If *universe* is provided only nodes within that universe are
        included (useful for scoped subgraph extraction).
        """
        _validate_node_id(root_id)
        visited: Set[str] = set()
        queue: List[Tuple[str, int]] = [(root_id, 0)]
        prefix = f"{universe.value}::" if universe else None

        while queue:
            current, depth = queue.pop(0)
            if current in visited:
                continue
            if prefix and not current.startswith(prefix):
                continue
            visited.add(current)
            if depth < max_depth:
                for successor in self._graph.successors(current):
                    queue.append((successor, depth + 1))

        sub = GraphStore()
        sub._graph = self._graph.subgraph(visited).copy()
        return sub

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the entire graph to a JSON-compatible dictionary."""
        return {
            "nodes": [
                {"id": nid, **dict(data)}
                for nid, data in self._graph.nodes(data=True)
            ],
            "edges": [
                {"source_id": u, "target_id": v, "relation": k, **dict(data)}
                for u, v, k, data in self._graph.edges(data=True, keys=True)
            ],
        }

    def save(self, path: str | Path) -> None:
        """Persist the graph as a JSON file at *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        logger.info("Graph saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "GraphStore":
        """Load a previously saved graph from *path*."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        store = cls()
        for node_data in data.get("nodes", []):
            node_type_value = node_data.get("node_type")
            # Resolve the concrete model class
            node_type = NodeType(node_type_value)
            node_cls = _NODE_TYPE_MAP.get(node_type, BaseNode)
            node = node_cls.model_validate(node_data)
            store._graph.add_node(node.id, **node_data)

        for edge_data in data.get("edges", []):
            src = edge_data["source_id"]
            tgt = edge_data["target_id"]
            rel = edge_data["relation"]
            store._graph.add_edge(src, tgt, key=rel, **edge_data)

        logger.info("Graph loaded from %s (%d nodes, %d edges)",
                    path, store._graph.number_of_nodes(), store._graph.number_of_edges())
        return store

    # ------------------------------------------------------------------
    # Convenience / statistics
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    def universe_stats(self) -> Dict[str, Dict[str, int]]:
        """Return node and edge counts per universe."""
        stats: Dict[str, Dict[str, int]] = {}
        for universe in Universe:
            prefix = f"{universe.value}::"
            node_count = sum(
                1 for nid in self._graph.nodes if nid.startswith(prefix)
            )
            edge_count = sum(
                1
                for u, v in self._graph.edges()
                if u.startswith(prefix) and v.startswith(prefix)
            )
            stats[universe.value] = {"nodes": node_count, "edges": edge_count}
        return stats

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GraphStore(nodes={self.node_count}, edges={self.edge_count})"
        )
