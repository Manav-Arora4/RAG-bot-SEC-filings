"""Embed retrieval chunks and persist them in a local Chroma collection."""

from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import (
    BGE_QUERY_PREFIX,
    CHROMA_COLLECTION_NAME,
    CHROMA_DIR,
    CHUNKS_PATH,
    DEFAULT_EMBED_BATCH_SIZE,
    EMBEDDING_MODEL_NAME,
)


METADATA_FIELDS = (
    "source_doc",
    "ticker",
    "company",
    "fiscal_year",
    "form",
    "accession_number",
    "filing_date",
    "report_date",
    "source_url",
    "chunk_index",
    "chunk_word_count",
)


def load_chunks(path: Path) -> list[dict[str, Any]]:
    """Load chunk records from the processed JSONL file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Chunks not found: {path}. Run `python -m src.chunk` first."
        )

    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    if not records:
        raise RuntimeError(f"No chunks found in {path}.")

    return records


def sanitize_metadata(record: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Return Chroma-safe metadata without null values."""
    metadata: dict[str, str | int | float | bool] = {}
    for field in METADATA_FIELDS:
        value = record.get(field)
        if value is not None:
            metadata[field] = value
    return metadata


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Load and cache the configured sentence-transformer model."""
    return SentenceTransformer(model_name)


def embed_query(model: SentenceTransformer, query: str) -> list[float]:
    """Embed a user query with the BGE retrieval prefix."""
    vector = model.encode(
        f"{BGE_QUERY_PREFIX}{query}",
        normalize_embeddings=True,
    )
    return vector.tolist()


def embed_passages(
    model: SentenceTransformer,
    texts: list[str],
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> list[list[float]]:
    """Embed document chunks for storage and cosine retrieval."""
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > batch_size,
        normalize_embeddings=True,
    )
    return vectors.tolist()


def get_chroma_client() -> chromadb.PersistentClient:
    """Return a persistent Chroma client rooted at data/chroma_db/."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_or_create_collection(
    client: chromadb.PersistentClient,
    *,
    reset: bool = False,
) -> Collection:
    """Create or reset the filings collection."""
    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
        except ValueError:
            pass

    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def get_indexed_collection(
    client: chromadb.PersistentClient | None = None,
) -> Collection:
    """Return the existing filings collection or raise a helpful error."""
    chroma_client = client or get_chroma_client()
    try:
        return chroma_client.get_collection(CHROMA_COLLECTION_NAME)
    except ValueError as exc:
        raise RuntimeError(
            f"Chroma collection '{CHROMA_COLLECTION_NAME}' not found. "
            "Run `python -m src.embed_store` first."
        ) from exc


def upsert_chunks(
    collection: Collection,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> None:
    """Write chunk embeddings and metadata into Chroma."""
    for start in tqdm(
        range(0, len(chunks), batch_size),
        desc="Upserting chunks",
        unit="batch",
    ):
        batch = chunks[start : start + batch_size]
        collection.upsert(
            ids=[record["chunk_id"] for record in batch],
            documents=[record["text"] for record in batch],
            embeddings=embeddings[start : start + batch_size],
            metadatas=[sanitize_metadata(record) for record in batch],
        )


def build_index(
    *,
    chunks_path: Path = CHUNKS_PATH,
    rebuild: bool = False,
    batch_size: int = DEFAULT_EMBED_BATCH_SIZE,
) -> int:
    """Embed all chunks and persist them to the local Chroma collection."""
    chunks = load_chunks(chunks_path)
    model = get_embedding_model()
    texts = [record["text"] for record in chunks]
    embeddings = embed_passages(model, texts, batch_size=batch_size)

    client = get_chroma_client()
    collection = get_or_create_collection(client, reset=rebuild)
    upsert_chunks(collection, chunks, embeddings, batch_size=batch_size)

    return collection.count()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed processed chunks and build the local Chroma index."
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=CHUNKS_PATH,
        help="Input JSONL path produced by src.chunk.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete and recreate the Chroma collection before upserting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EMBED_BATCH_SIZE,
        help="Batch size for embedding and Chroma upserts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    count = build_index(
        chunks_path=args.chunks,
        rebuild=args.rebuild,
        batch_size=args.batch_size,
    )
    print(f"Indexed {count} chunks in {CHROMA_DIR}")


if __name__ == "__main__":
    main()
