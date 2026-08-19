"""Generate citation-grounded answers from retrieved filing chunks."""

from __future__ import annotations

import argparse
import re
import sys
from functools import lru_cache
from typing import Any

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from groq import Groq

from src.config import (
    DEFAULT_GROQ_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_TOP_K,
    get_required_env,
)
from src.retrieve import retrieve_chunks

CITATION_PATTERN = re.compile(r"[\[【](\d+)[\]】]")
SYSTEM_PROMPT = (
    "You are a financial analyst assistant. Answer the user's question using ONLY "
    "the provided SEC filing excerpts. If the excerpts do not contain enough "
    "information, say you cannot determine the answer from the provided context. "
    "Cite supporting statements with bracketed source numbers like [1] or [2]. "
    "Be concise and precise with numbers when they appear in the context."
)


@lru_cache(maxsize=1)
def get_groq_client() -> Groq:
    """Return a cached Groq client."""
    return Groq(api_key=get_required_env("GROQ_API_KEY"))


def format_citation(chunk: dict[str, Any], index: int) -> str:
    """Build a human-readable citation string for one retrieved chunk."""
    metadata = chunk.get("metadata", {})
    ticker = metadata.get("ticker", "UNKNOWN")
    company = metadata.get("company", ticker)
    fiscal_year = metadata.get("fiscal_year", "unknown")
    chunk_id = chunk.get("chunk_id", "unknown")
    source_url = metadata.get("source_url")
    label = f"[{index}] {company} ({ticker}) FY{fiscal_year} 10-K ({chunk_id})"
    if source_url:
        return f"{label} — {source_url}"
    return label


def build_context(retrieved_chunks: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into numbered context blocks for the LLM."""
    blocks: list[str] = []
    for index, chunk in enumerate(retrieved_chunks, start=1):
        metadata = chunk.get("metadata", {})
        header = (
            f"[{index}] {metadata.get('company', metadata.get('ticker', 'UNKNOWN'))} "
            f"({metadata.get('ticker', 'UNKNOWN')}) "
            f"FY{metadata.get('fiscal_year', 'unknown')} "
            f"({chunk.get('chunk_id', 'unknown')})"
        )
        blocks.append(f"{header}\n{chunk.get('text', '').strip()}")
    return "\n\n".join(blocks)


def extract_cited_indices(answer: str) -> list[int]:
    """Return unique 1-based citation indices referenced in the answer."""
    indices: list[int] = []
    for match in CITATION_PATTERN.finditer(answer):
        index = int(match.group(1))
        if index not in indices:
            indices.append(index)
    return indices


def build_citations(
    answer: str,
    retrieved_chunks: list[dict[str, Any]],
) -> list[str]:
    """Map bracket citations in the answer to formatted source strings."""
    cited_indices = extract_cited_indices(answer)
    if not cited_indices:
        cited_indices = list(range(1, len(retrieved_chunks) + 1))

    citations: list[str] = []
    for index in cited_indices:
        if 1 <= index <= len(retrieved_chunks):
            citations.append(format_citation(retrieved_chunks[index - 1], index))
    return citations


def generate_answer(
    question: str,
    retrieved_chunks: list[dict[str, Any]],
    *,
    model: str = DEFAULT_GROQ_MODEL,
    provider: str = DEFAULT_LLM_PROVIDER,
) -> dict[str, Any]:
    """Call the configured LLM and return a grounded answer with citations."""
    if not question.strip():
        raise ValueError("question must not be empty")
    if not retrieved_chunks:
        return {
            "answer": "No relevant filing excerpts were retrieved for this question.",
            "citations": [],
            "cited_chunk_ids": [],
        }
    if provider != "groq":
        raise RuntimeError(f"Unsupported LLM provider: {provider}")

    context = build_context(retrieved_chunks)
    client = get_groq_client()

    response = None
    for attempt in range(6):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Context:\n{context}\n\n"
                            f"Question: {question.strip()}\n\n"
                            "Answer:"
                        ),
                    },
                ],
                temperature=0.1,
            )
            break
        except Exception as exc:
            # Catch rate limit errors and retry with backoff
            error_str = str(exc).lower()
            if "rate_limit" in error_str or "429" in error_str or "tpm" in error_str:
                wait_time = 2.0 * (attempt + 1)
                import time
                time.sleep(wait_time)
            else:
                raise

    if response is None:
        raise RuntimeError("Failed to generate answer after retries.")

    answer = response.choices[0].message.content or ""
    answer = answer.strip()
    citations = build_citations(answer, retrieved_chunks)
    cited_chunk_ids = [
        retrieved_chunks[index - 1]["chunk_id"]
        for index in extract_cited_indices(answer)
        if 1 <= index <= len(retrieved_chunks)
    ]

    return {
        "answer": answer,
        "citations": citations,
        "cited_chunk_ids": cited_chunk_ids,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve filing context and generate a cited answer."
    )
    parser.add_argument("question", help="Question to answer.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of chunks to retrieve before generation.",
    )
    parser.add_argument(
        "--ticker",
        help="Optional ticker filter, for example AAPL.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_GROQ_MODEL,
        help="Groq model name.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retrieved_chunks = retrieve_chunks(
        args.question,
        top_k=args.top_k,
        ticker=args.ticker,
    )
    result = generate_answer(
        args.question,
        retrieved_chunks,
        model=args.model,
    )

    print(result["answer"])
    if result["citations"]:
        print("\nCitations:")
        for citation in result["citations"]:
            print(f"- {citation}")


if __name__ == "__main__":
    main()
