"""End-to-end RAG pipeline."""

from __future__ import annotations

from typing import Any

from src.config import DEFAULT_GROQ_MODEL, DEFAULT_TOP_K
from src.generate import generate_answer
from src.retrieve import retrieve_chunks


def answer_query(
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    ticker: str | None = None,
    model: str = DEFAULT_GROQ_MODEL,
) -> dict[str, Any]:
    """Retrieve filing evidence and generate a cited answer."""
    retrieved_chunks = retrieve_chunks(question, top_k=top_k, ticker=ticker)
    generation = generate_answer(
        question,
        retrieved_chunks,
        model=model,
    )

    return {
        "question": question,
        "answer": generation["answer"],
        "citations": generation["citations"],
        "cited_chunk_ids": generation.get("cited_chunk_ids", []),
        "retrieved_chunks": retrieved_chunks,
    }
