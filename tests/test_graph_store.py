"""
Tests for animelore/graph_store.py

Covers:
- Node insertion and retrieval
- Edge insertion and retrieval
- Cross-universe edge prevention (enforced by Relationship model)
- Universe-scoped node queries
- Multi-hop path finding
- BFS subgraph extraction
- Serialisation round-trip
- Universe stats
"""
import json
import tempfile
from pathlib import Path

import pytest

from animelore.graph_store import GraphStore
from animelore.models import (
    CharacterNode,
    EventNode,
    FactionNode,
    AbilityNode,
    LocationNode,
    NodeType,
    Relationship,
    RelationType,
    Universe,
    build_node_id,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_character(universe: Universe, name: str) -> CharacterNode:
    return CharacterNode(id=build_node_id(universe, name), name=name)


def _make_faction(universe: Universe, name: str) -> FactionNode:
    return FactionNode(id=build_node_id(universe, name), name=name)


def _make_event(universe: Universe, name: str) -> EventNode:
    return EventNode(id=build_node_id(universe, name), name=name)


def _make_location(universe: Universe, name: str) -> LocationNode:
    return LocationNode(id=build_node_id(universe, name), name=name)


def _make_ability(universe: Universe, name: str) -> AbilityNode:
    return AbilityNode(id=build_node_id(universe, name), name=name)


@pytest.fixture()
def store() -> GraphStore:
    return GraphStore()


@pytest.fixture()
def populated_store() -> GraphStore:
    """
    A GraphStore pre-loaded with a small One Piece subgraph:
      luffy --MEMBER_OF--> straw_hat_pirates
      luffy --ALLIED_WITH--> zoro
      luffy --PARTICIPATED_IN--> marineford_war
      marineford_war --OCCURRED_AT--> marineford
    """
    gs = GraphStore()
    luffy = _make_character(Universe.one_piece, "luffy")
    zoro = _make_character(Universe.one_piece, "zoro")
    shp = _make_faction(Universe.one_piece, "straw hat pirates")
    war = _make_event(Universe.one_piece, "marineford war")
    loc = _make_location(Universe.one_piece, "marineford")

    for node in [luffy, zoro, shp, war, loc]:
        gs.add_node(node)

    gs.add_edge(Relationship(source_id=luffy.id, target_id=shp.id, relation=RelationType.member_of))
    gs.add_edge(Relationship(source_id=luffy.id, target_id=zoro.id, relation=RelationType.allied_with))
    gs.add_edge(Relationship(source_id=luffy.id, target_id=war.id, relation=RelationType.participated_in))
    gs.add_edge(Relationship(source_id=war.id, target_id=loc.id, relation=RelationType.occurred_at))

    return gs


# ---------------------------------------------------------------------------
# Node operations
# ---------------------------------------------------------------------------

class TestNodeOperations:
    def test_add_and_get(self, store):
        node = _make_character(Universe.one_piece, "luffy")
        store.add_node(node)
        data = store.get_node(node.id)
        assert data is not None
        assert data["name"] == "luffy"

    def test_get_missing(self, store):
        assert store.get_node("one_piece::nobody") is None

    def test_has_node(self, store):
        node = _make_character(Universe.jujutsu_kaisen, "gojo_satoru")
        assert not store.has_node(node.id)
        store.add_node(node)
        assert store.has_node(node.id)

    def test_merge_on_duplicate(self, store):
        """Adding the same node ID twice merges (does not overwrite) attributes."""
        node = CharacterNode(
            id="one_piece::luffy",
            name="luffy",
            description="Captain",
        )
        store.add_node(node)
        # Add again with new property — should not overwrite existing
        node2 = CharacterNode(
            id="one_piece::luffy",
            name="luffy",
            description="New description",
        )
        store.add_node(node2)
        # The first description is kept (merge preserves existing keys)
        data = store.get_node("one_piece::luffy")
        assert data["description"] == "Captain"

    def test_nodes_by_universe(self, populated_store):
        one_piece_nodes = populated_store.nodes_by_universe(Universe.one_piece)
        assert len(one_piece_nodes) == 5

        jjk_nodes = populated_store.nodes_by_universe(Universe.jujutsu_kaisen)
        assert len(jjk_nodes) == 0

    def test_nodes_by_type(self, populated_store):
        characters = populated_store.nodes_by_type(NodeType.character)
        assert len(characters) == 2

        factions = populated_store.nodes_by_type(NodeType.faction)
        assert len(factions) == 1

    def test_nodes_by_type_and_universe(self, store):
        store.add_node(_make_character(Universe.one_piece, "luffy"))
        store.add_node(_make_character(Universe.jujutsu_kaisen, "gojo"))
        chars = store.nodes_by_type(NodeType.character, universe=Universe.one_piece)
        assert len(chars) == 1
        assert chars[0]["name"] == "luffy"


# ---------------------------------------------------------------------------
# Edge operations
# ---------------------------------------------------------------------------

class TestEdgeOperations:
    def test_add_and_get(self, store):
        a = _make_character(Universe.one_piece, "luffy")
        b = _make_character(Universe.one_piece, "zoro")
        store.add_node(a)
        store.add_node(b)
        rel = Relationship(source_id=a.id, target_id=b.id, relation=RelationType.allied_with)
        store.add_edge(rel)

        edges = store.get_edges(source_id=a.id)
        assert len(edges) == 1
        assert edges[0]["relation"] == RelationType.allied_with.value

    def test_missing_source_node(self, store):
        b = _make_character(Universe.one_piece, "zoro")
        store.add_node(b)
        rel = Relationship(
            source_id="one_piece::nobody",
            target_id=b.id,
            relation=RelationType.allied_with,
        )
        with pytest.raises(KeyError, match="Source node"):
            store.add_edge(rel)

    def test_missing_target_node(self, store):
        a = _make_character(Universe.one_piece, "luffy")
        store.add_node(a)
        rel = Relationship(
            source_id=a.id,
            target_id="one_piece::nobody",
            relation=RelationType.allied_with,
        )
        with pytest.raises(KeyError, match="Target node"):
            store.add_edge(rel)

    def test_get_edges_by_relation(self, populated_store):
        edges = populated_store.get_edges(relation=RelationType.member_of)
        assert len(edges) == 1

    def test_get_all_edges(self, populated_store):
        all_edges = populated_store.get_edges()
        assert len(all_edges) == 4  # fixture has 4 edges

    def test_edge_count(self, populated_store):
        assert populated_store.edge_count == 4


# ---------------------------------------------------------------------------
# Multi-hop traversal
# ---------------------------------------------------------------------------

class TestTraversal:
    def test_neighbors_outbound(self, populated_store):
        luffy_id = build_node_id(Universe.one_piece, "luffy")
        neighbors = populated_store.neighbors(luffy_id)
        # luffy → shp, zoro, marineford_war
        assert len(neighbors) == 3

    def test_neighbors_filtered_by_relation(self, populated_store):
        luffy_id = build_node_id(Universe.one_piece, "luffy")
        allied = populated_store.neighbors(luffy_id, relation=RelationType.allied_with)
        assert len(allied) == 1
        assert "zoro" in allied[0]

    def test_multi_hop_paths(self, populated_store):
        luffy_id = build_node_id(Universe.one_piece, "luffy")
        marineford_id = build_node_id(Universe.one_piece, "marineford")
        # luffy -> marineford_war -> marineford (2 hops)
        paths = populated_store.multi_hop_paths(luffy_id, marineford_id, max_hops=3)
        assert len(paths) >= 1
        assert marineford_id in paths[0]

    def test_no_path_found(self, populated_store):
        luffy_id = build_node_id(Universe.one_piece, "luffy")
        shp_id = build_node_id(Universe.one_piece, "straw hat pirates")
        # There's no reverse path from shp back to luffy with default direction
        paths = populated_store.multi_hop_paths(shp_id, luffy_id, max_hops=2)
        assert paths == []

    def test_bfs_subgraph(self, populated_store):
        luffy_id = build_node_id(Universe.one_piece, "luffy")
        sub = populated_store.bfs_subgraph(luffy_id, max_depth=1)
        # luffy + direct neighbours (shp, zoro, marineford_war)
        assert sub.node_count >= 4

    def test_bfs_subgraph_universe_scoped(self, store):
        # Add nodes from two universes
        luffy = _make_character(Universe.one_piece, "luffy")
        gojo = _make_character(Universe.jujutsu_kaisen, "gojo")
        store.add_node(luffy)
        store.add_node(gojo)
        # BFS scoped to one_piece: should never include gojo
        sub = store.bfs_subgraph(luffy.id, max_depth=2, universe=Universe.one_piece)
        node_ids = {n["id"] for n in sub.to_dict()["nodes"]}
        assert gojo.id not in node_ids


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_round_trip(self, populated_store):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.json"
            populated_store.save(path)
            assert path.exists()

            loaded = GraphStore.load(path)
            assert loaded.node_count == populated_store.node_count
            assert loaded.edge_count == populated_store.edge_count

    def test_json_structure(self, populated_store):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "graph.json"
            populated_store.save(path)
            with open(path) as f:
                data = json.load(f)
            assert "nodes" in data
            assert "edges" in data
            assert len(data["nodes"]) == populated_store.node_count

    def test_to_dict(self, populated_store):
        d = populated_store.to_dict()
        assert isinstance(d["nodes"], list)
        assert isinstance(d["edges"], list)


# ---------------------------------------------------------------------------
# Universe stats
# ---------------------------------------------------------------------------

class TestUniverseStats:
    def test_stats(self, populated_store):
        stats = populated_store.universe_stats()
        assert stats["one_piece"]["nodes"] == 5
        assert stats["one_piece"]["edges"] == 4
        assert stats["jujutsu_kaisen"]["nodes"] == 0

    def test_empty_store(self, store):
        stats = store.universe_stats()
        for u in ["one_piece", "jujutsu_kaisen", "one_punch_man"]:
            assert stats[u]["nodes"] == 0
            assert stats[u]["edges"] == 0
