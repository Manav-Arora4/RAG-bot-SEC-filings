"""Retrieve top-k filing chunks for a user query."""

from __future__ import annotations

import argparse
import textwrap
from typing import Any

from src.config import DEFAULT_TOP_K
from src.embed_store import embed_query, get_embedding_model, get_indexed_collection


def cosine_similarity_from_distance(distance: float) -> float:
    """Convert Chroma cosine distance to a similarity score."""
    return 1.0 - distance


def format_retrieved_chunk(
    *,
    chunk_id: str,
    text: str,
    distance: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Normalize one Chroma hit into a pipeline-friendly record."""
    return {
        "chunk_id": chunk_id,
        "text": text,
        "distance": distance,
        "similarity": cosine_similarity_from_distance(distance),
        "metadata": metadata,
    }


def retrieve_chunks(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    ticker: str | None = None,
) -> list[dict[str, Any]]:
    """Embed a query and return the most relevant filing chunks."""
    if not query.strip():
        raise ValueError("query must not be empty")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    model = get_embedding_model()
    query_embedding = embed_query(model, query.strip())
    collection = get_indexed_collection()

    where = {"ticker": ticker.upper()} if ticker else None
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved: list[dict[str, Any]] = []
    for chunk_id, text, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
        strict=True,
    ):
        retrieved.append(
            format_retrieved_chunk(
                chunk_id=chunk_id,
                text=text or "",
                distance=distance,
                metadata=metadata or {},
            )
        )

    return retrieved


def format_source_label(chunk: dict[str, Any]) -> str:
    """Build a short human-readable source label for CLI output."""
    metadata = chunk.get("metadata", {})
    ticker = metadata.get("ticker", "UNKNOWN")
    fiscal_year = metadata.get("fiscal_year", "unknown")
    chunk_index = metadata.get("chunk_index", "?")
    similarity = chunk.get("similarity", 0.0)
    return (
        f"{ticker} FY{fiscal_year} chunk {chunk_index} "
        f"(similarity {similarity:.3f}, id={chunk['chunk_id']})"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve top-k filing chunks for a natural-language query."
    )
    parser.add_argument("query", help="Question or search query.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve.",
    )
    parser.add_argument(
        "--ticker",
        help="Optional ticker filter, for example AAPL.",
    )
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=240,
        help="Characters of chunk text to print per result.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = retrieve_chunks(args.query, top_k=args.top_k, ticker=args.ticker)

    print(f"Query: {args.query}")
    print(f"Retrieved {len(chunks)} chunk(s)\n")

    for rank, chunk in enumerate(chunks, start=1):
        preview = textwrap.shorten(
            chunk["text"].replace("\n", " "),
            width=args.preview_chars,
            placeholder="...",
        )
        print(f"{rank}. {format_source_label(chunk)}")
        print(f"   {preview}\n")


if __name__ == "__main__":
    main()
