"""
Tests for animelore/models.py

Covers:
- Namespace-protected ID validation
- build_node_id helper
- All concrete node type instantiations
- Relationship model (valid and invalid cross-universe)
- ExtractionResult schema
"""
import pytest
from pydantic import ValidationError

from animelore.models import (
    AbilityNode,
    CharacterNode,
    EventNode,
    ExtractionResult,
    ExtractedTriplet,
    FactionNode,
    LocationNode,
    NodeType,
    Relationship,
    RelationType,
    Universe,
    build_node_id,
    _validate_node_id,
)


# ---------------------------------------------------------------------------
# build_node_id
# ---------------------------------------------------------------------------

class TestBuildNodeId:
    def test_basic(self):
        nid = build_node_id(Universe.one_piece, "Monkey D. Luffy")
        assert nid == "one_piece::monkey_d_luffy"

    def test_underscore_normalisation(self):
        nid = build_node_id(Universe.jujutsu_kaisen, "Ryomen Sukuna")
        assert nid == "jujutsu_kaisen::ryomen_sukuna"

    def test_string_universe(self):
        nid = build_node_id("one_punch_man", "Saitama")
        assert nid == "one_punch_man::saitama"

    def test_special_chars_stripped(self):
        nid = build_node_id(Universe.one_piece, "Roronoa Zoro!")
        assert nid == "one_piece::roronoa_zoro"

    def test_numbers_preserved(self):
        nid = build_node_id(Universe.one_punch_man, "S-Class Hero")
        assert nid == "one_punch_man::s_class_hero"


# ---------------------------------------------------------------------------
# _validate_node_id
# ---------------------------------------------------------------------------

class TestValidateNodeId:
    def test_valid_ids(self):
        valid = [
            "one_piece::luffy",
            "jujutsu_kaisen::gojo_satoru",
            "one_punch_man::saitama",
            "one_piece::monkey_d_luffy",
            "one_punch_man::s_class_hero_1",
        ]
        for v in valid:
            assert _validate_node_id(v) == v

    def test_invalid_no_separator(self):
        with pytest.raises(ValueError, match="does not match"):
            _validate_node_id("luffy")

    def test_invalid_wrong_universe(self):
        with pytest.raises(ValueError, match="does not match"):
            _validate_node_id("naruto::naruto_uzumaki")

    def test_invalid_uppercase(self):
        with pytest.raises(ValueError, match="does not match"):
            _validate_node_id("one_piece::Luffy")

    def test_invalid_empty_name(self):
        with pytest.raises(ValueError, match="does not match"):
            _validate_node_id("one_piece::")

    def test_invalid_name_starts_with_underscore(self):
        with pytest.raises(ValueError, match="does not match"):
            _validate_node_id("one_piece::_luffy")


# ---------------------------------------------------------------------------
# Node models
# ---------------------------------------------------------------------------

class TestCharacterNode:
    def test_valid(self):
        node = CharacterNode(
            id="one_piece::luffy",
            name="Monkey D. Luffy",
            description="Captain of the Straw Hat Pirates",
        )
        assert node.node_type == NodeType.character
        assert node.universe == Universe.one_piece

    def test_invalid_id(self):
        with pytest.raises(ValidationError):
            CharacterNode(id="invalid", name="Luffy")

    def test_aliases(self):
        node = CharacterNode(
            id="one_piece::luffy",
            name="Monkey D. Luffy",
            aliases=["Straw Hat Luffy", "Captain Luffy"],
        )
        assert len(node.aliases) == 2

    def test_universe_property(self):
        node = CharacterNode(id="jujutsu_kaisen::gojo_satoru", name="Gojo Satoru")
        assert node.universe == Universe.jujutsu_kaisen


class TestFactionNode:
    def test_valid(self):
        node = FactionNode(id="one_piece::straw_hat_pirates", name="Straw Hat Pirates")
        assert node.node_type == NodeType.faction


class TestAbilityNode:
    def test_valid(self):
        node = AbilityNode(id="jujutsu_kaisen::infinity", name="Infinity")
        assert node.node_type == NodeType.ability


class TestLocationNode:
    def test_valid(self):
        node = LocationNode(id="jujutsu_kaisen::shibuya", name="Shibuya")
        assert node.node_type == NodeType.location


class TestEventNode:
    def test_valid(self):
        node = EventNode(
            id="one_piece::marineford_war",
            name="Marineford War",
        )
        assert node.node_type == NodeType.event


# ---------------------------------------------------------------------------
# Relationship model
# ---------------------------------------------------------------------------

class TestRelationship:
    def _make_rel(self, src, tgt, rel=RelationType.allied_with):
        return Relationship(source_id=src, target_id=tgt, relation=rel)

    def test_valid_same_universe(self):
        rel = self._make_rel("one_piece::luffy", "one_piece::zoro")
        assert rel.source_universe == Universe.one_piece
        assert rel.target_universe == Universe.one_piece

    def test_cross_universe_forbidden(self):
        with pytest.raises(ValidationError, match="Cross-universe"):
            self._make_rel("one_piece::luffy", "jujutsu_kaisen::gojo_satoru")

    def test_invalid_source_id(self):
        with pytest.raises(ValidationError):
            self._make_rel("luffy", "one_piece::zoro")

    def test_invalid_target_id(self):
        with pytest.raises(ValidationError):
            self._make_rel("one_piece::luffy", "zoro")

    def test_all_relation_types(self):
        pairs = [
            ("one_piece::luffy", "one_piece::straw_hat_pirates", RelationType.member_of),
            ("one_piece::luffy", "one_piece::zoro", RelationType.allied_with),
            ("one_piece::luffy", "one_piece::blackbeard", RelationType.fought),
            ("one_piece::luffy", "one_piece::gomu_gomu_no_mi", RelationType.wields),
            ("one_piece::marineford_war", "one_piece::marineford", RelationType.occurred_at),
            ("one_piece::luffy", "one_piece::marineford_war", RelationType.participated_in),
        ]
        for src, tgt, rel in pairs:
            r = Relationship(source_id=src, target_id=tgt, relation=rel)
            assert r.relation == rel

    def test_properties_stored(self):
        rel = self._make_rel(
            "one_piece::luffy", "one_piece::zoro",
            RelationType.allied_with,
        )
        # Build with properties
        rel2 = Relationship(
            source_id="one_piece::luffy",
            target_id="one_piece::zoro",
            relation=RelationType.allied_with,
            properties={"since": "Syrup Village"},
        )
        assert rel2.properties["since"] == "Syrup Village"


# ---------------------------------------------------------------------------
# ExtractionResult
# ---------------------------------------------------------------------------

class TestExtractionResult:
    def test_valid(self):
        triplet = ExtractedTriplet(
            subject_name="Luffy",
            subject_type=NodeType.character,
            relation=RelationType.member_of,
            object_name="Straw Hat Pirates",
            object_type=NodeType.faction,
            confidence=0.95,
        )
        result = ExtractionResult(
            universe=Universe.one_piece,
            source_text="Luffy is the captain of the Straw Hat Pirates.",
            triplets=[triplet],
        )
        assert result.universe == Universe.one_piece
        assert len(result.triplets) == 1
        assert result.triplets[0].confidence == 0.95

    def test_empty_triplets(self):
        result = ExtractionResult(
            universe=Universe.jujutsu_kaisen,
            source_text="No entities here.",
        )
        assert result.triplets == []

    def test_invalid_universe(self):
        with pytest.raises(ValidationError):
            ExtractionResult(
                universe="dragon_ball",  # type: ignore[arg-type]
                source_text="test",
            )

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            ExtractedTriplet(
                subject_name="X",
                subject_type=NodeType.character,
                relation=RelationType.fought,
                object_name="Y",
                object_type=NodeType.character,
                confidence=1.5,  # > 1.0 — invalid
            )
        with pytest.raises(ValidationError):
            ExtractedTriplet(
                subject_name="X",
                subject_type=NodeType.character,
                relation=RelationType.fought,
                object_name="Y",
                object_type=NodeType.character,
                confidence=-0.1,  # < 0.0 — invalid
            )
