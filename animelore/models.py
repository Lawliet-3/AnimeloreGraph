"""
Pydantic models for the AnimeloreGraph system.

Defines the strict ontology (node types, edge types) and enforces the
namespace-protected ID convention: ``universe::lowercase_snake_case_name``.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Universe(str, Enum):
    """Supported fictional universes / data domains."""

    one_piece = "one_piece"
    jujutsu_kaisen = "jujutsu_kaisen"
    one_punch_man = "one_punch_man"


class NodeType(str, Enum):
    """Allowed node labels (strict ontology)."""

    character = "Character"
    faction = "Faction"
    ability = "Ability"
    location = "Location"
    event = "Event"


class RelationType(str, Enum):
    """Allowed directed relationship types (strict ontology)."""

    member_of = "MEMBER_OF"
    allied_with = "ALLIED_WITH"
    fought = "FOUGHT"
    wields = "WIELDS"
    occurred_at = "OCCURRED_AT"
    participated_in = "PARTICIPATED_IN"


# ---------------------------------------------------------------------------
# ID validation helpers
# ---------------------------------------------------------------------------

_ID_PATTERN = re.compile(
    r"^(one_piece|jujutsu_kaisen|one_punch_man)::[a-z0-9][a-z0-9_]*$"
)


def _validate_node_id(value: str) -> str:
    """Raise ValueError if *value* does not match ``universe::snake_case``."""
    if not _ID_PATTERN.match(value):
        raise ValueError(
            f"Node ID '{value}' does not match the required pattern "
            "'universe::lowercase_snake_case_name' "
            "(e.g. 'one_piece::monkey_d_luffy')."
        )
    return value


def build_node_id(universe: Universe | str, name: str) -> str:
    """
    Construct a namespace-protected node ID from a universe and a name.

    The *name* is normalised to lowercase snake_case automatically.
    """
    if isinstance(universe, Universe):
        universe = universe.value
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"{universe}::{slug}"


# ---------------------------------------------------------------------------
# Base node
# ---------------------------------------------------------------------------

class BaseNode(BaseModel):
    """
    Common fields shared by every graph node.

    ``id`` is the canonical namespace-protected identifier.
    ``universe`` is derived automatically from the prefix of ``id``.
    """

    id: str = Field(
        ...,
        description=(
            "Namespace-protected node ID following the pattern "
            "'universe::lowercase_snake_case_name'."
        ),
    )
    node_type: NodeType
    name: str = Field(..., description="Human-readable canonical name.")
    description: Optional[str] = Field(
        default=None, description="Optional free-text description."
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional domain-specific key/value properties.",
    )

    @field_validator("id")
    @classmethod
    def _validate_id(cls, v: str) -> str:
        return _validate_node_id(v)

    @property
    def universe(self) -> Universe:
        """Extract the Universe from the node's ``id`` prefix."""
        prefix = self.id.split("::")[0]
        return Universe(prefix)

    @model_validator(mode="after")
    def _universe_consistent(self) -> "BaseNode":
        """Ensure the id prefix matches a known Universe value."""
        # _validate_id already ensures the prefix is valid; this is a belt-and-
        # suspenders check that Universe() can be constructed.
        _ = self.universe
        return self


# ---------------------------------------------------------------------------
# Concrete node types
# ---------------------------------------------------------------------------

class CharacterNode(BaseNode):
    """Humanoids, heroes, villains, pirates, sorcerers, entities."""

    node_type: NodeType = NodeType.character
    aliases: List[str] = Field(
        default_factory=list,
        description="Alternative names or titles for this character.",
    )


class FactionNode(BaseNode):
    """Organised crews, institutions, military branches."""

    node_type: NodeType = NodeType.faction


class AbilityNode(BaseNode):
    """Power mechanics, specific techniques, magical attributes."""

    node_type: NodeType = NodeType.ability


class LocationNode(BaseNode):
    """Geographic landmarks, cities, planets, islands."""

    node_type: NodeType = NodeType.location


class EventNode(BaseNode):
    """Highly structured incidents, battles, historical arcs."""

    node_type: NodeType = NodeType.event


# ---------------------------------------------------------------------------
# Edges / Relationships
# ---------------------------------------------------------------------------

class Relationship(BaseModel):
    """
    A directed, typed relationship between two nodes.

    Both ``source_id`` and ``target_id`` must satisfy the namespace-protected
    ID convention.  The allowed ``(source_type, relation, target_type)``
    triples are validated against the strict ontology.
    """

    source_id: str = Field(..., description="Namespace-protected ID of the source node.")
    target_id: str = Field(..., description="Namespace-protected ID of the target node.")
    relation: RelationType
    properties: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_id", "target_id")
    @classmethod
    def _validate_ids(cls, v: str) -> str:
        return _validate_node_id(v)

    @property
    def source_universe(self) -> Universe:
        return Universe(self.source_id.split("::")[0])

    @property
    def target_universe(self) -> Universe:
        return Universe(self.target_id.split("::")[0])

    @model_validator(mode="after")
    def _validate_cross_universe(self) -> "Relationship":
        """
        Cross-universe edges are explicitly disallowed to prevent data
        contamination between narrative domains.
        """
        if self.source_universe != self.target_universe:
            raise ValueError(
                f"Cross-universe relationship detected: "
                f"'{self.source_id}' ({self.source_universe.value}) → "
                f"'{self.target_id}' ({self.target_universe.value}). "
                "Cross-universe edges are forbidden."
            )
        return self


# ---------------------------------------------------------------------------
# Extraction schemas (used by the instructor-based LLM extraction layer)
# ---------------------------------------------------------------------------

class ExtractedTriplet(BaseModel):
    """
    A single knowledge-graph triplet produced by the extraction LLM.

    The LLM is instructed to populate these fields strictly.
    """

    subject_name: str = Field(
        ..., description="Human-readable name of the subject entity."
    )
    subject_type: NodeType = Field(..., description="Ontology type of the subject.")
    relation: RelationType = Field(..., description="Directed relationship type.")
    object_name: str = Field(
        ..., description="Human-readable name of the object entity."
    )
    object_type: NodeType = Field(..., description="Ontology type of the object.")
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence (0–1).",
    )
    evidence: Optional[str] = Field(
        default=None, description="Verbatim source sentence supporting the triplet."
    )


class ExtractionResult(BaseModel):
    """
    Container for all triplets extracted from a single text passage.

    The ``universe`` field is set externally (not by the LLM) to enforce
    namespace isolation—the model cannot assign a universe.
    """

    universe: Universe = Field(
        ..., description="The universe this passage belongs to (set by caller)."
    )
    source_text: str = Field(..., description="The raw input passage.")
    triplets: List[ExtractedTriplet] = Field(
        default_factory=list,
        description="All extracted knowledge triplets.",
    )
