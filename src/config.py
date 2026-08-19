"""Shared configuration helpers for FinRAG-Mini."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHUNKS_PATH = PROCESSED_DATA_DIR / "chunks.jsonl"
CHROMA_DIR = DATA_DIR / "chroma_db"
FILINGS_MANIFEST_PATH = DATA_DIR / "filings_manifest.json"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHROMA_COLLECTION_NAME = "filings"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_EMBED_BATCH_SIZE = 64
DEFAULT_TOP_K = 5
DEFAULT_LLM_PROVIDER = "groq"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_SEC_USER_AGENT = "finrag-mini/0.1 contact@example.com"


def load_environment() -> None:
    """Load local environment variables from .env when present."""
    load_dotenv(PROJECT_ROOT / ".env")


def get_required_env(name: str) -> str:
    """Return a required environment variable or raise a clear error."""
    load_environment()
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing {name}. Add it to .env or export it in your shell."
        )
    return value
