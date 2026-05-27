"""
Tests for animelore/query_engine.py

Covers:
- find_paths: multi-hop path discovery
- get_character_connections: BFS subgraph
- find_shared_events
- find_characters_in_faction
- find_abilities_of_character
- semantic_search (mocked embed_fn + InMemoryVectorStore)
- aggregate_stats
- find_highly_connected_nodes
- Universe scoping (no cross-universe contamination)
"""
import pytest

from animelore.embeddings import InMemoryVectorStore
from animelore.graph_store import GraphStore
from animelore.models import (
    AbilityNode,
    CharacterNode,
    EventNode,
    FactionNode,
    LocationNode,
    NodeType,
    Relationship,
    RelationType,
    Universe,
    build_node_id,
)
from animelore.query_engine import AggregateResult, PathResult, QueryEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ch(universe, name):
    return CharacterNode(id=build_node_id(universe, name), name=name)


def _faction(universe, name):
    return FactionNode(id=build_node_id(universe, name), name=name)


def _event(universe, name):
    return EventNode(id=build_node_id(universe, name), name=name)


def _loc(universe, name):
    return LocationNode(id=build_node_id(universe, name), name=name)


def _ab(universe, name):
    return AbilityNode(id=build_node_id(universe, name), name=name)


def _rel(src_id, tgt_id, rtype):
    return Relationship(source_id=src_id, target_id=tgt_id, relation=rtype)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def graph_and_engine():
    """
    One Piece knowledge graph:
      luffy --MEMBER_OF--> straw_hat_pirates
      zoro  --MEMBER_OF--> straw_hat_pirates
      luffy --ALLIED_WITH--> zoro
      luffy --WIELDS--> gomu_gomu
      luffy --PARTICIPATED_IN--> marineford_war
      zoro  --PARTICIPATED_IN--> marineford_war
      marineford_war --OCCURRED_AT--> marineford
    """
    gs = GraphStore()
    OP = Universe.one_piece

    luffy = _ch(OP, "luffy")
    zoro = _ch(OP, "zoro")
    shp = _faction(OP, "straw hat pirates")
    gomu = _ab(OP, "gomu gomu")
    war = _event(OP, "marineford war")
    marineford = _loc(OP, "marineford")

    for node in [luffy, zoro, shp, gomu, war, marineford]:
        gs.add_node(node)

    gs.add_edge(_rel(luffy.id, shp.id, RelationType.member_of))
    gs.add_edge(_rel(zoro.id, shp.id, RelationType.member_of))
    gs.add_edge(_rel(luffy.id, zoro.id, RelationType.allied_with))
    gs.add_edge(_rel(luffy.id, gomu.id, RelationType.wields))
    gs.add_edge(_rel(luffy.id, war.id, RelationType.participated_in))
    gs.add_edge(_rel(zoro.id, war.id, RelationType.participated_in))
    gs.add_edge(_rel(war.id, marineford.id, RelationType.occurred_at))

    engine = QueryEngine(graph_store=gs)
    return gs, engine


# ---------------------------------------------------------------------------
# find_paths
# ---------------------------------------------------------------------------

class TestFindPaths:
    def test_direct_path(self, graph_and_engine):
        gs, engine = graph_and_engine
        luffy_id = build_node_id(Universe.one_piece, "luffy")
        shp_id = build_node_id(Universe.one_piece, "straw hat pirates")
        results = engine.find_paths(luffy_id, shp_id, max_hops=2)
        assert len(results) >= 1
        assert isinstance(results[0], PathResult)
        assert results[0].hops == 1

    def test_two_hop_path(self, graph_and_engine):
        _gs, engine = graph_and_engine
        luffy_id = build_node_id(Universe.one_piece, "luffy")
        marineford_id = build_node_id(Universe.one_piece, "marineford")
        results = engine.find_paths(luffy_id, marineford_id, max_hops=3)
        assert len(results) >= 1
        assert results[0].hops == 2

    def test_no_path(self, graph_and_engine):
        _gs, engine = graph_and_engine
        shp_id = build_node_id(Universe.one_piece, "straw hat pirates")
        luffy_id = build_node_id(Universe.one_piece, "luffy")
        # No reverse path from shp to luffy
        results = engine.find_paths(shp_id, luffy_id, max_hops=3)
        assert results == []

    def test_sorted_by_hops(self, graph_and_engine):
        _gs, engine = graph_and_engine
        luffy_id = build_node_id(Universe.one_piece, "luffy")
        marineford_id = build_node_id(Universe.one_piece, "marineford")
        results = engine.find_paths(luffy_id, marineford_id, max_hops=5)
        hops = [r.hops for r in results]
        assert hops == sorted(hops)

    def test_missing_node_returns_empty(self, graph_and_engine):
        _gs, engine = graph_and_engine
        results = engine.find_paths(
            "one_piece::nobody",
            "one_piece::luffy",
            max_hops=2,
        )
        assert results == []


# ---------------------------------------------------------------------------
# get_character_connections
# ---------------------------------------------------------------------------

class TestCharacterConnections:
    def test_returns_subgraph_dict(self, graph_and_engine):
        _gs, engine = graph_and_engine
        luffy_id = build_node_id(Universe.one_piece, "luffy")
        result = engine.get_character_connections(luffy_id, depth=1)
        assert "nodes" in result
        assert "edges" in result

    def test_scoped_to_universe(self, graph_and_engine):
        gs, engine = graph_and_engine
        # Add a JJK node — should not appear in luffy's subgraph
        gojo = _ch(Universe.jujutsu_kaisen, "gojo")
        gs.add_node(gojo)
        luffy_id = build_node_id(Universe.one_piece, "luffy")
        result = engine.get_character_connections(luffy_id, depth=2)
        node_ids = {n["id"] for n in result["nodes"]}
        assert gojo.id not in node_ids

    def test_invalid_universe_raises(self, graph_and_engine):
        _gs, engine = graph_and_engine
        with pytest.raises(ValueError):
            engine.get_character_connections("unknown::luffy", depth=1)


# ---------------------------------------------------------------------------
# find_shared_events
# ---------------------------------------------------------------------------

class TestFindSharedEvents:
    def test_shared_event(self, graph_and_engine):
        _gs, engine = graph_and_engine
        luffy_id = build_node_id(Universe.one_piece, "luffy")
        zoro_id = build_node_id(Universe.one_piece, "zoro")
        shared = engine.find_shared_events(luffy_id, zoro_id)
        assert len(shared) == 1
        assert "marineford_war" in shared[0]

    def test_no_shared_events(self, graph_and_engine):
        _gs, engine = graph_and_engine
        # zoro has no ally with gomu gomu relation
        zoro_id = build_node_id(Universe.one_piece, "zoro")
        gomu_id = build_node_id(Universe.one_piece, "gomu gomu")
        shared = engine.find_shared_events(zoro_id, gomu_id)
        assert shared == []


# ---------------------------------------------------------------------------
# find_characters_in_faction
# ---------------------------------------------------------------------------

class TestFindCharactersInFaction:
    def test_members(self, graph_and_engine):
        _gs, engine = graph_and_engine
        shp_id = build_node_id(Universe.one_piece, "straw hat pirates")
        members = engine.find_characters_in_faction(shp_id)
        assert len(members) == 2
        member_ids = set(members)
        assert build_node_id(Universe.one_piece, "luffy") in member_ids
        assert build_node_id(Universe.one_piece, "zoro") in member_ids

    def test_empty_faction(self, graph_and_engine):
        gs, engine = graph_and_engine
        empty_faction = _faction(Universe.one_piece, "empty faction")
        gs.add_node(empty_faction)
        members = engine.find_characters_in_faction(empty_faction.id)
        assert members == []


# ---------------------------------------------------------------------------
# find_abilities_of_character
# ---------------------------------------------------------------------------

class TestFindAbilitiesOfCharacter:
    def test_abilities(self, graph_and_engine):
        _gs, engine = graph_and_engine
        luffy_id = build_node_id(Universe.one_piece, "luffy")
        abilities = engine.find_abilities_of_character(luffy_id)
        assert len(abilities) == 1
        assert "gomu_gomu" in abilities[0]

    def test_no_abilities(self, graph_and_engine):
        _gs, engine = graph_and_engine
        zoro_id = build_node_id(Universe.one_piece, "zoro")
        abilities = engine.find_abilities_of_character(zoro_id)
        assert abilities == []


# ---------------------------------------------------------------------------
# semantic_search
# ---------------------------------------------------------------------------

class TestSemanticSearch:
    def _make_engine_with_vectors(self, graph_and_engine):
        gs, _engine = graph_and_engine
        vector_store = InMemoryVectorStore()

        # Simple 3-D mock embeddings
        embeddings = {
            build_node_id(Universe.one_piece, "luffy"): [1.0, 0.0, 0.0],
            build_node_id(Universe.one_piece, "zoro"): [0.0, 1.0, 0.0],
            build_node_id(Universe.one_piece, "straw hat pirates"): [0.5, 0.5, 0.0],
        }
        for nid, vec in embeddings.items():
            vector_store.upsert(nid, vec)

        embed_fn = lambda text: [1.0, 0.0, 0.0]  # always returns "luffy-like" vector
        engine = QueryEngine(graph_store=gs, vector_store=vector_store, embed_fn=embed_fn)
        return engine

    def test_semantic_search_returns_results(self, graph_and_engine):
        engine = self._make_engine_with_vectors(graph_and_engine)
        results = engine.semantic_search("pirate captain", top_k=2)
        assert len(results) <= 2
        assert all(hasattr(r, "node_id") for r in results)
        assert all(hasattr(r, "score") for r in results)

    def test_semantic_search_universe_filter(self, graph_and_engine):
        gs, _engine = graph_and_engine
        vector_store = InMemoryVectorStore()

        op_luffy_id = build_node_id(Universe.one_piece, "luffy")
        jjk_gojo = _ch(Universe.jujutsu_kaisen, "gojo")
        gs.add_node(jjk_gojo)
        jjk_gojo_id = jjk_gojo.id

        vector_store.upsert(op_luffy_id, [1.0, 0.0])
        vector_store.upsert(jjk_gojo_id, [0.9, 0.1])

        embed_fn = lambda text: [1.0, 0.0]
        engine = QueryEngine(graph_store=gs, vector_store=vector_store, embed_fn=embed_fn)

        results = engine.semantic_search("fighter", universe=Universe.one_piece)
        result_ids = {r.node_id for r in results}
        assert jjk_gojo_id not in result_ids, "JJK node leaked into One Piece semantic search"

    def test_semantic_search_raises_without_vector_store(self, graph_and_engine):
        _gs, engine = graph_and_engine
        with pytest.raises(RuntimeError, match="VectorStore"):
            engine.semantic_search("test")


# ---------------------------------------------------------------------------
# aggregate_stats
# ---------------------------------------------------------------------------

class TestAggregateStats:
    def test_all_universes(self, graph_and_engine):
        _gs, engine = graph_and_engine
        stats = engine.aggregate_stats()
        assert len(stats) == 3  # one_piece, jujutsu_kaisen, one_punch_man
        one_piece_stat = next(s for s in stats if s.universe == "one_piece")
        assert one_piece_stat.total_nodes == 6
        assert one_piece_stat.total_edges == 7

    def test_single_universe(self, graph_and_engine):
        _gs, engine = graph_and_engine
        stats = engine.aggregate_stats(universe=Universe.one_piece)
        assert len(stats) == 1
        assert stats[0].universe == "one_piece"

    def test_node_type_counts(self, graph_and_engine):
        _gs, engine = graph_and_engine
        stats = engine.aggregate_stats(universe=Universe.one_piece)
        counts = stats[0].node_type_counts
        assert counts.get("Character", 0) == 2
        assert counts.get("Faction", 0) == 1

    def test_empty_universe_stats(self, graph_and_engine):
        _gs, engine = graph_and_engine
        stats = engine.aggregate_stats(universe=Universe.jujutsu_kaisen)
        assert stats[0].total_nodes == 0


# ---------------------------------------------------------------------------
# find_highly_connected_nodes
# ---------------------------------------------------------------------------

class TestHighlyConnectedNodes:
    def test_luffy_is_top(self, graph_and_engine):
        _gs, engine = graph_and_engine
        top = engine.find_highly_connected_nodes(universe=Universe.one_piece, top_k=3)
        assert len(top) <= 3
        top_ids = [nid for nid, _ in top]
        # Luffy has 5 outbound edges and is likely the most connected
        assert build_node_id(Universe.one_piece, "luffy") in top_ids

    def test_universe_scoped(self, graph_and_engine):
        gs, engine = graph_and_engine
        gojo = _ch(Universe.jujutsu_kaisen, "gojo")
        gs.add_node(gojo)
        top = engine.find_highly_connected_nodes(universe=Universe.one_piece)
        ids = [nid for nid, _ in top]
        assert gojo.id not in ids
