"""Run a local in-memory semantic search demo."""
from __future__ import annotations

import os

from openai import OpenAI

from animelore import AnimeloreGraphPipeline, Universe


def _get_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is not set. Export it to run this demo.")
    return api_key


def main() -> None:
    api_key = _get_api_key()
    client = OpenAI(api_key=api_key)

    def embed_fn(text: str) -> list[float]:
        return client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        ).data[0].embedding

    pipeline = AnimeloreGraphPipeline(
        openai_api_key=api_key,
        embed_fn=embed_fn,
    )

    pipeline.ingest(
        text="Monkey D. Luffy is the captain of the Straw Hat Pirates.",
        universe=Universe.one_piece,
        auto_index=True,
    )
    pipeline.ingest(
        text="Roronoa Zoro is a swordsman in the Straw Hat Pirates.",
        universe=Universe.one_piece,
        auto_index=True,
    )
    pipeline.ingest(
        text="Monkey D. Luffy fought Buggy.",
        universe=Universe.one_piece,
        auto_index=True,
    )

    results = pipeline.query_semantic(
        query_text="straw hat captain",
        top_k=5,
        universe=Universe.one_piece,
    )

    print("Semantic hits:")
    for hit in results:
        print(f"- {hit.node_id} (score={hit.score:.3f})")


if __name__ == "__main__":
    main()
