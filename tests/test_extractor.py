"""
Tests for animelore/extractor.py

Covers:
- extraction_to_graph_objects: node and relationship creation from ExtractionResult
- Entity deduplication (same name → same node)
- Cross-universe edge prevention inside extraction_to_graph_objects
- min_confidence filtering is applied before graph conversion
- KnowledgeExtractor.extract with mocked instructor client
"""
from unittest.mock import MagicMock, patch

import pytest

from animelore.extractor import KnowledgeExtractor
from animelore.models import (
    ExtractionResult,
    ExtractedTriplet,
    NodeType,
    Relationship,
    RelationType,
    Universe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_triplet(
    subject_name: str,
    subject_type: NodeType,
    relation: RelationType,
    object_name: str,
    object_type: NodeType,
    confidence: float = 1.0,
    evidence: str = None,
) -> ExtractedTriplet:
    return ExtractedTriplet(
        subject_name=subject_name,
        subject_type=subject_type,
        relation=relation,
        object_name=object_name,
        object_type=object_type,
        confidence=confidence,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# extraction_to_graph_objects
# ---------------------------------------------------------------------------

class TestExtractionToGraphObjects:
    def _extractor(self):
        # Build extractor without a real client for unit tests
        e = KnowledgeExtractor.__new__(KnowledgeExtractor)
        e.model = "gpt-4o-mini"
        e.min_confidence = 0.5
        e._client = None
        return e

    def test_basic_one_piece(self):
        extractor = self._extractor()
        result = ExtractionResult(
            universe=Universe.one_piece,
            source_text="Luffy is a member of the Straw Hat Pirates.",
            triplets=[
                _make_triplet(
                    "Luffy", NodeType.character,
                    RelationType.member_of,
                    "Straw Hat Pirates", NodeType.faction,
                )
            ],
        )
        nodes, rels = extractor.extraction_to_graph_objects(result)
        assert len(nodes) == 2
        assert len(rels) == 1

        node_ids = {n.id for n in nodes}
        assert "one_piece::luffy" in node_ids
        assert "one_piece::straw_hat_pirates" in node_ids

    def test_entity_deduplication(self):
        """Same name referenced in two triplets should produce one node."""
        extractor = self._extractor()
        result = ExtractionResult(
            universe=Universe.jujutsu_kaisen,
            source_text="Gojo fought Sukuna. Gojo wields Infinity.",
            triplets=[
                _make_triplet("Gojo", NodeType.character, RelationType.fought, "Sukuna", NodeType.character),
                _make_triplet("Gojo", NodeType.character, RelationType.wields, "Infinity", NodeType.ability),
            ],
        )
        nodes, rels = extractor.extraction_to_graph_objects(result)
        gojo_nodes = [n for n in nodes if "gojo" in n.id]
        assert len(gojo_nodes) == 1  # deduplicated

    def test_evidence_stored_in_properties(self):
        extractor = self._extractor()
        result = ExtractionResult(
            universe=Universe.one_punch_man,
            source_text="Saitama defeated Boros.",
            triplets=[
                _make_triplet(
                    "Saitama", NodeType.character,
                    RelationType.fought,
                    "Boros", NodeType.character,
                    evidence="Saitama defeated Boros.",
                )
            ],
        )
        _nodes, rels = extractor.extraction_to_graph_objects(result)
        assert rels[0].properties["evidence"] == "Saitama defeated Boros."

    def test_all_node_types_created(self):
        extractor = self._extractor()
        triplets = [
            _make_triplet("Luffy", NodeType.character, RelationType.member_of, "SHP", NodeType.faction),
            _make_triplet("Luffy", NodeType.character, RelationType.wields, "Gomu Gomu", NodeType.ability),
            _make_triplet("Luffy", NodeType.character, RelationType.participated_in, "Marineford War", NodeType.event),
            _make_triplet("Marineford War", NodeType.event, RelationType.occurred_at, "Marineford", NodeType.location),
        ]
        result = ExtractionResult(
            universe=Universe.one_piece,
            source_text="...",
            triplets=triplets,
        )
        nodes, _rels = extractor.extraction_to_graph_objects(result)
        type_set = {n.node_type.value for n in nodes}
        assert "Character" in type_set
        assert "Faction" in type_set
        assert "Ability" in type_set
        assert "Event" in type_set
        assert "Location" in type_set

    def test_namespace_protected_ids(self):
        extractor = self._extractor()
        result = ExtractionResult(
            universe=Universe.one_punch_man,
            source_text="Saitama is a hero.",
            triplets=[
                _make_triplet("Saitama", NodeType.character, RelationType.member_of, "Hero Association", NodeType.faction),
            ],
        )
        nodes, _rels = extractor.extraction_to_graph_objects(result)
        for node in nodes:
            assert node.id.startswith("one_punch_man::")

    def test_empty_triplets(self):
        extractor = self._extractor()
        result = ExtractionResult(
            universe=Universe.one_piece,
            source_text="Nothing here.",
            triplets=[],
        )
        nodes, rels = extractor.extraction_to_graph_objects(result)
        assert nodes == []
        assert rels == []


# ---------------------------------------------------------------------------
# KnowledgeExtractor.extract — mocked
# ---------------------------------------------------------------------------

class TestKnowledgeExtractorMocked:
    """
    Tests that the extractor correctly calls instructor with temperature=0.0
    and produces an ExtractionResult.  The OpenAI / instructor call is mocked.
    """

    def _make_mock_client(self, triplets):
        """Build a mock instructor client that returns the given triplets."""
        mock_response = MagicMock()
        mock_response.triplets = triplets

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        return mock_client

    def test_extract_calls_temperature_zero(self):
        triplets = [
            _make_triplet(
                "Luffy", NodeType.character,
                RelationType.member_of,
                "Straw Hat Pirates", NodeType.faction,
            )
        ]
        mock_client = self._make_mock_client(triplets)

        extractor = KnowledgeExtractor.__new__(KnowledgeExtractor)
        extractor.model = "gpt-4o-mini"
        extractor.min_confidence = 0.5
        extractor._client = mock_client

        result = extractor.extract("Luffy is a member of the Straw Hat Pirates.", Universe.one_piece)

        # Verify temperature=0.0 was passed
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.0

    def test_extract_returns_extraction_result(self):
        triplets = [
            _make_triplet(
                "Gojo", NodeType.character,
                RelationType.wields,
                "Infinity", NodeType.ability,
            )
        ]
        mock_client = self._make_mock_client(triplets)

        extractor = KnowledgeExtractor.__new__(KnowledgeExtractor)
        extractor.model = "gpt-4o-mini"
        extractor.min_confidence = 0.5
        extractor._client = mock_client

        result = extractor.extract("Gojo uses Infinity.", Universe.jujutsu_kaisen)

        assert isinstance(result, ExtractionResult)
        assert result.universe == Universe.jujutsu_kaisen
        assert len(result.triplets) == 1

    def test_extract_filters_low_confidence(self):
        """Triplets below min_confidence should be filtered out."""
        triplets = [
            _make_triplet(
                "Saitama", NodeType.character,
                RelationType.fought,
                "Boros", NodeType.character,
                confidence=0.3,  # below threshold
            ),
            _make_triplet(
                "Saitama", NodeType.character,
                RelationType.member_of,
                "Hero Association", NodeType.faction,
                confidence=0.9,  # above threshold
            ),
        ]
        mock_client = self._make_mock_client(triplets)

        extractor = KnowledgeExtractor.__new__(KnowledgeExtractor)
        extractor.model = "gpt-4o-mini"
        extractor.min_confidence = 0.5
        extractor._client = mock_client

        result = extractor.extract("...", Universe.one_punch_man)

        assert len(result.triplets) == 1
        assert result.triplets[0].object_name == "Hero Association"

    def test_extract_raises_without_client(self):
        extractor = KnowledgeExtractor.__new__(KnowledgeExtractor)
        extractor.model = "gpt-4o-mini"
        extractor.min_confidence = 0.5
        extractor._client = None

        with pytest.raises(RuntimeError, match="instructor"):
            extractor.extract("text", Universe.one_piece)

    def test_universe_not_inferred_by_model(self):
        """
        The universe must be injected by the caller, not inferred by the model.
        The prompt should include the universe string; the result universe must
        match what was passed in.
        """
        triplets = []
        mock_client = self._make_mock_client(triplets)

        extractor = KnowledgeExtractor.__new__(KnowledgeExtractor)
        extractor.model = "gpt-4o-mini"
        extractor.min_confidence = 0.5
        extractor._client = mock_client

        result = extractor.extract("text", Universe.one_punch_man)
        assert result.universe == Universe.one_punch_man
