"""Split cleaned filings into metadata-rich retrieval chunks."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

from src.config import CHUNKS_PATH, FILINGS_MANIFEST_PATH, PROCESSED_DATA_DIR, PROJECT_ROOT


DEFAULT_CHUNK_WORDS = 650
DEFAULT_OVERLAP_WORDS = 100
MIN_PARAGRAPH_WORDS = 20


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Load filing records created by src.ingest."""
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}. Run `python -m src.ingest` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(path_value: str) -> Path:
    """Resolve a manifest path relative to the project root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def word_count(text: str) -> int:
    """Count word-like tokens for rough chunk sizing."""
    return len(re.findall(r"\S+", text))


def paragraph_blocks(text: str) -> list[str]:
    """Group cleaned text into paragraph-ish blocks for chunk assembly."""
    raw_blocks = re.split(r"\n\s*\n", normalize_text(text))
    blocks: list[str] = []

    for raw_block in raw_blocks:
        lines = [line.strip() for line in raw_block.splitlines() if line.strip()]
        if not lines:
            continue

        merged_lines: list[str] = []
        buffer: list[str] = []
        for line in lines:
            if word_count(line) < MIN_PARAGRAPH_WORDS:
                buffer.append(line)
                continue

            if buffer:
                merged_lines.append(" ".join(buffer))
                buffer = []
            merged_lines.append(line)

        if buffer:
            merged_lines.append(" ".join(buffer))

        block = "\n".join(merged_lines).strip()
        if block:
            blocks.append(block)

    return blocks


def split_long_block(block: str, chunk_words: int, overlap_words: int) -> list[str]:
    """Split one oversized block into overlapping word windows."""
    words = block.split()
    if len(words) <= chunk_words:
        return [block]

    step = chunk_words - overlap_words
    if step <= 0:
        raise ValueError("overlap_words must be smaller than chunk_words")

    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_words, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += step
    return chunks


def trailing_overlap(text: str, overlap_words: int) -> str:
    """Return the trailing words from a chunk for overlap with the next chunk."""
    if overlap_words <= 0:
        return ""
    words = text.split()
    return " ".join(words[-overlap_words:])


def split_into_chunks(
    text: str,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    overlap_words: int = DEFAULT_OVERLAP_WORDS,
) -> list[str]:
    """Split text into paragraph-aware overlapping chunks."""
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    chunks: list[str] = []
    current_parts: list[str] = []
    current_words = 0

    for block in paragraph_blocks(text):
        candidate_blocks = split_long_block(block, chunk_words, overlap_words)

        for candidate in candidate_blocks:
            candidate_words = word_count(candidate)
            if current_parts and current_words + candidate_words > chunk_words:
                chunk_text = "\n\n".join(current_parts).strip()
                chunks.append(chunk_text)

                overlap = trailing_overlap(chunk_text, overlap_words)
                current_parts = [overlap] if overlap else []
                current_words = word_count(overlap) if overlap else 0

            current_parts.append(candidate)
            current_words += candidate_words

    if current_parts:
        chunks.append("\n\n".join(current_parts).strip())

    return [chunk for chunk in chunks if chunk]


def source_doc_id(record: dict[str, Any]) -> str:
    """Return a stable source document identifier."""
    ticker = str(record["ticker"]).upper()
    fiscal_year = record.get("fiscal_year") or str(record.get("report_date", ""))[:4]
    return f"{ticker}_{fiscal_year}"


def build_chunk_records(
    manifest_records: Iterable[dict[str, Any]],
    chunk_words: int,
    overlap_words: int,
) -> list[dict[str, Any]]:
    """Build JSON-serializable chunk records from manifest entries."""
    chunk_records: list[dict[str, Any]] = []

    for record in manifest_records:
        input_path = resolve_project_path(record["local_path"])
        if not input_path.exists():
            raise FileNotFoundError(f"Raw filing not found: {input_path}")

        source_doc = source_doc_id(record)
        text = input_path.read_text(encoding="utf-8", errors="ignore")
        chunks = split_into_chunks(text, chunk_words=chunk_words, overlap_words=overlap_words)

        for index, chunk_text in enumerate(chunks):
            chunk_records.append(
                {
                    "chunk_id": f"{source_doc}_{index:04d}",
                    "source_doc": source_doc,
                    "ticker": record["ticker"],
                    "company": record["company"],
                    "fiscal_year": record.get("fiscal_year"),
                    "form": record.get("form"),
                    "accession_number": record.get("accession_number"),
                    "filing_date": record.get("filing_date"),
                    "report_date": record.get("report_date"),
                    "source_url": record.get("source_url"),
                    "chunk_index": index,
                    "chunk_word_count": word_count(chunk_text),
                    "text": chunk_text,
                }
            )

        print(f"Chunked {source_doc}: {len(chunks)} chunks")

    return chunk_records


def write_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> int:
    """Write records to a JSONL file and return the row count."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=True) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split cleaned filings into metadata-rich JSONL chunks."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=FILINGS_MANIFEST_PATH,
        help="Path to filings_manifest.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CHUNKS_PATH,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=DEFAULT_CHUNK_WORDS,
        help="Approximate words per chunk.",
    )
    parser.add_argument(
        "--overlap-words",
        type=int,
        default=DEFAULT_OVERLAP_WORDS,
        help="Words repeated from the previous chunk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    records = build_chunk_records(
        manifest,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
    )
    count = write_jsonl(records, args.output)
    print(f"Wrote {count} chunks to {args.output}")


if __name__ == "__main__":
    main()
