# FinRAG-Mini

A compact Retrieval-Augmented Generation system for answering questions over public company financial filings with cited evidence and retrieval evaluation.

## Current Status

Phases 1-6 are in place:
- **Ingestion & Cleaning**: Downloads and parses 2023 10-K filings from SEC EDGAR.
- **Chunking**: Splits filings into metadata-rich JSONL chunks (709 chunks).
- **Embeddings & Vector Store**: Dense indexing with `BAAI/bge-small-en-v1.5` in local ChromaDB.
- **Retrieval**: Top-k semantic vector search with optional ticker filtering.
- **Generation & Citations**: LLM response generation with bracketed citations (`[1]`, `[2]`).
- **Evaluation & Benchmarking**: Automated evaluation suite calculating Hit Rate@K, MRR, Context Keyword Recall, Factual Grounding, and Citation Validity.

## Planned Architecture

```text
SEC Filing -> Clean Text -> Chunks -> Embeddings -> Chroma
                                             |
User Question -> Query Embedding -> Retrieval -> LLM -> Cited Answer
                                                     |
                                            Automated Evaluation
                                          (eval/run_eval.py)
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create `.env` from `.env.example` and provide a Groq API key:

```env
GROQ_API_KEY=your_groq_api_key_here
SEC_USER_AGENT=finrag-mini/0.1 your_email@example.com
```

## Usage

Download and clean the default SEC filings:

```bash
python -m src.ingest
```

Or clean manually downloaded filings from a local folder:

```bash
python -m src.ingest --manual-dir path\to\filings
```

Split cleaned filings into retrieval chunks:

```bash
python -m src.chunk
```

Embed chunks and build the Chroma index:

```bash
python -m src.embed_store
```

Use `--rebuild` to recreate the collection from scratch.

Retrieve top-k chunks for a query:

```bash
python -m src.retrieve "What were Apple's net sales in fiscal 2023?"
```

Ask a question and get a cited answer:

```bash
python query.py "What were Apple's net sales in fiscal 2023?"
```

Or run retrieval and generation directly:

```bash
python -m src.generate "What were Apple's net sales in fiscal 2023?"
```

## Running Evaluation & Benchmarks

Run the automated evaluation benchmark across 16 curated test questions covering factual financial metrics, strategic risks, cross-company comparisons, and negative refusal tests:

```bash
# Fast offline retrieval-only evaluation (Hit Rate @ K, MRR, Context Recall)
python -m eval.run_eval --skip-generation

# Full end-to-end evaluation with LLM grounding and citation checks
python -m eval.run_eval
```

Evaluation outputs:
- **`eval/eval_report.md`**: Executive scorecard, category breakdown, and detailed per-question tables.
- **`eval/eval_results.json`**: Machine-readable evaluation dataset.

## Current Corpus

The default ingestion command downloads 2023 10-K filings for:

- AAPL (Apple Inc.)
- MSFT (Microsoft Corp.)
- JPM (JPMorgan Chase & Co.)
- TSLA (Tesla, Inc.)
- WMT (Walmart Inc.)

The chunking pass creates 709 chunks across the five filings using roughly 650 words per chunk with 100 words of overlap.
