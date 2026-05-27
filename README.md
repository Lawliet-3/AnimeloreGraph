# AnimeloreGraph

**Multi-Universe Anime Lore GraphRAG System**

An advanced Knowledge Graph Augmented Generation (GraphRAG) engine designed for deep multi-hop reasoning, semantic search, and aggregate global queries across multiple complex, narrative-heavy fictional domains.

Initial reference datasets: **One Piece**, **Jujutsu Kaisen**, **One Punch Man**.

---

## Architecture

```
animelore/
├── models.py         # Pydantic ontology: nodes, edges, namespace-protected IDs
├── graph_store.py    # NetworkX MultiDiGraph with serialisation
├── extractor.py      # instructor + OpenAI extraction layer (temperature=0.0)
├── embeddings.py     # VectorStore abstraction (in-memory, Qdrant)
├── query_engine.py   # Multi-hop traversal, semantic search, aggregate queries
└── pipeline.py       # End-to-end AnimeloreGraphPipeline orchestrator
tests/
├── test_models.py
├── test_graph_store.py
├── test_extractor.py
└── test_query_engine.py
```

## Strict Ontology

### Node Types
| Label | Description |
|-------|-------------|
| `Character` | Humanoids, heroes, villains, pirates, sorcerers |
| `Faction` | Organised crews, institutions, military branches |
| `Ability` | Power mechanics, techniques, magical attributes |
| `Location` | Geographic landmarks, cities, islands |
| `Event` | Battles, historical arcs, structured incidents |

### Edge Types
| Relationship | Direction |
|-------------|-----------|
| `MEMBER_OF` | `(:Character)→(:Faction)` |
| `ALLIED_WITH` | `(:Character)→(:Character)` |
| `FOUGHT` | `(:Character)→(:Character)` |
| `WIELDS` | `(:Character)→(:Ability)` |
| `OCCURRED_AT` | `(:Event)→(:Location)` |
| `PARTICIPATED_IN` | `(:Character)→(:Event)` |

## Namespace Protection

Every node ID follows the strict pattern `universe::lowercase_snake_case_name`:

```
one_piece::monkey_d_luffy
jujutsu_kaisen::gojo_satoru
one_punch_man::saitama
```

Cross-universe edges are **hard-rejected** by the `Relationship` Pydantic validator. The LLM extraction layer never infers or assigns a universe — it is always set by the caller.

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies (no external DB required for basic usage):
- `networkx` — in-memory graph
- `pydantic` — schema validation
- `instructor` + `openai` — LLM extraction
- `qdrant-client` — optional vector store backend

## Usage

```python
from animelore import AnimeloreGraphPipeline, Universe

pipeline = AnimeloreGraphPipeline(openai_api_key="sk-...")

# Ingest a passage
pipeline.ingest(
    text="Monkey D. Luffy is the captain of the Straw Hat Pirates.",
    universe=Universe.one_piece,
)

# Multi-hop path query
paths = pipeline.query_paths(
    source_id="one_piece::monkey_d_luffy",
    target_id="one_piece::marineford",
    max_hops=3,
)

# Aggregate stats
stats = pipeline.query_aggregate()

# Save / load
pipeline.save("knowledge_graph.json")
pipeline2 = AnimeloreGraphPipeline.load("knowledge_graph.json")
```

## Tests

```bash
pytest tests/ -v
```
