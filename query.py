"""CLI entrypoint for FinRAG-Mini."""

from __future__ import annotations

import argparse
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.pipeline import answer_query


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question over financial filings.")
    parser.add_argument("question", help="Question to answer using the filing corpus.")
    args = parser.parse_args()

    result = answer_query(args.question)
    print(result["answer"])

    if result.get("citations"):
        print("\nCitations:")
        for citation in result["citations"]:
            print(f"- {citation}")


if __name__ == "__main__":
    main()

