"""
AnimeloreGraph — Multi-Universe Anime Lore GraphRAG System.

Public API surface:
- ``AnimeloreGraphPipeline``: end-to-end ingest + query orchestrator.
- ``GraphStore``: NetworkX-backed knowledge graph.
- ``QueryEngine``: multi-hop traversal and semantic search.
- ``KnowledgeExtractor``: LLM-powered triplet extraction.
- ``VectorStore`` / ``InMemoryVectorStore``: embeddings abstraction.
- ``models.*``: Pydantic schema definitions.
"""

from .embeddings import InMemoryVectorStore, VectorStore, create_vector_store
from .extractor import KnowledgeExtractor
from .graph_store import GraphStore
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
from .pipeline import AnimeloreGraphPipeline
from .query_engine import AggregateResult, PathResult, QueryEngine, SemanticResult

__all__ = [
    # Pipeline
    "AnimeloreGraphPipeline",
    # Core components
    "GraphStore",
    "QueryEngine",
    "KnowledgeExtractor",
    # Embeddings
    "VectorStore",
    "InMemoryVectorStore",
    "create_vector_store",
    # Models
    "Universe",
    "NodeType",
    "RelationType",
    "BaseNode",
    "CharacterNode",
    "FactionNode",
    "AbilityNode",
    "LocationNode",
    "EventNode",
    "Relationship",
    "ExtractedTriplet",
    "ExtractionResult",
    "build_node_id",
    # Query result types
    "PathResult",
    "SemanticResult",
    "AggregateResult",
]
