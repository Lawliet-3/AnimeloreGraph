"""
LLM-powered knowledge extraction layer.

Uses the ``instructor`` library on top of the OpenAI client to perform
deterministic (``temperature=0.0``), schema-validated extraction of
knowledge-graph triplets from raw text.

The ``universe`` is **always** set by the caller — the LLM is never
permitted to assign or infer it, preserving strict namespace isolation.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from pydantic import BaseModel, Field

from .models import (
    AbilityNode,
    BaseNode,
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
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Instructor / OpenAI integration
# ---------------------------------------------------------------------------

try:
    import instructor
    from openai import OpenAI

    _INSTRUCTOR_AVAILABLE = True
except ImportError:  # pragma: no cover
    _INSTRUCTOR_AVAILABLE = False


_SYSTEM_PROMPT = """\
You are a precise knowledge-graph extraction assistant.
Given a passage from a fictional universe, extract ALL factual relationships
as structured triplets.  You MUST:
- Only use the allowed NodeTypes: Character, Faction, Ability, Location, Event.
- Only use the allowed RelationTypes: MEMBER_OF, ALLIED_WITH, FOUGHT, WIELDS,
  OCCURRED_AT, PARTICIPATED_IN.
- Never invent new node or relationship types.
- Set confidence to a value between 0.0 and 1.0.
- Populate evidence with the verbatim sentence that supports each triplet.
Return ONLY valid JSON matching the schema — no prose.
"""

_USER_PROMPT_TEMPLATE = """\
Universe: {universe}

Passage:
\"\"\"
{text}
\"\"\"

Extract all knowledge-graph triplets from the passage above.
"""


class KnowledgeExtractor:
    """
    Extracts structured ``ExtractionResult`` objects from raw text passages
    using an OpenAI model via the ``instructor`` library.

    Parameters
    ----------
    model:
        OpenAI model name.  Defaults to ``gpt-4o-mini`` as specified in the
        project requirements.
    api_key:
        OpenAI API key.  If ``None`` the environment variable
        ``OPENAI_API_KEY`` is used.
    min_confidence:
        Triplets with a confidence score below this threshold are filtered out.
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        min_confidence: float = 0.5,
    ) -> None:
        self.model = model
        self.min_confidence = min_confidence
        self._client = self._build_client(api_key)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_client(api_key: Optional[str]) -> Optional[Any]:
        """
        Build the instructor-patched OpenAI client.

        Returns ``None`` when instructor / openai are not installed so that
        tests can mock the extraction path without real dependencies.
        """
        if not _INSTRUCTOR_AVAILABLE:
            logger.warning(
                "instructor / openai packages are not installed.  "
                "KnowledgeExtractor will operate in stub mode only."
            )
            return None
        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        raw_client = OpenAI(**kwargs)
        return instructor.from_openai(raw_client)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        text: str,
        universe: Universe,
    ) -> ExtractionResult:
        """
        Extract knowledge-graph triplets from *text* for the given *universe*.

        The LLM call uses ``temperature=0.0`` for deterministic output.
        Cross-universe contamination is prevented because the ``universe``
        argument is injected by the caller and never inferred by the model.

        Parameters
        ----------
        text:
            Raw passage text to extract from.
        universe:
            The universe this passage belongs to.

        Returns
        -------
        ExtractionResult
            Validated and filtered extraction result.
        """
        if self._client is None:
            raise RuntimeError(
                "KnowledgeExtractor requires 'instructor' and 'openai' packages. "
                "Install them with: pip install instructor openai"
            )

        class _TripletList(BaseModel):
            triplets: List[ExtractedTriplet] = Field(default_factory=list)

        prompt = _USER_PROMPT_TEMPLATE.format(
            universe=universe.value,
            text=text,
        )

        response: _TripletList = self._client.chat.completions.create(
            model=self.model,
            response_model=_TripletList,
            temperature=0.0,  # deterministic extraction
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        filtered = [
            t for t in response.triplets if t.confidence >= self.min_confidence
        ]

        return ExtractionResult(
            universe=universe,
            source_text=text,
            triplets=filtered,
        )

    def extraction_to_graph_objects(
        self,
        result: ExtractionResult,
    ) -> tuple[list[BaseNode], list[Relationship]]:
        """
        Convert an ``ExtractionResult`` into concrete ``BaseNode`` instances
        and ``Relationship`` objects ready for insertion into a ``GraphStore``.

        Entity resolution: if the same name appears multiple times, the same
        node ID is reused, preventing duplicate nodes.
        """
        universe = result.universe
        nodes: dict[str, BaseNode] = {}
        relationships: list[Relationship] = []

        _type_to_cls = {
            NodeType.character: CharacterNode,
            NodeType.faction: FactionNode,
            NodeType.ability: AbilityNode,
            NodeType.location: LocationNode,
            NodeType.event: EventNode,
        }

        def _get_or_create(name: str, node_type: NodeType) -> str:
            nid = build_node_id(universe, name)
            if nid not in nodes:
                node_cls = _type_to_cls[node_type]
                nodes[nid] = node_cls(id=nid, name=name)
            return nid

        for triplet in result.triplets:
            src_id = _get_or_create(triplet.subject_name, triplet.subject_type)
            tgt_id = _get_or_create(triplet.object_name, triplet.object_type)
            try:
                rel = Relationship(
                    source_id=src_id,
                    target_id=tgt_id,
                    relation=triplet.relation,
                    properties=(
                        {"evidence": triplet.evidence} if triplet.evidence else {}
                    ),
                )
                relationships.append(rel)
            except ValueError as exc:
                logger.warning("Skipping invalid relationship: %s", exc)

        return list(nodes.values()), relationships
