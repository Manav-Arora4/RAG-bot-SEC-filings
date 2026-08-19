# FinRAG-Mini

A Retrieval-Augmented Generation (RAG) system for querying and extracting verified financial data from SEC Form 10-K annual filings. The system implements automated EDGAR ingestion, document cleaning, metadata-preserving chunking, dense vector indexing, cited answer generation, and a quantitative evaluation framework.

---

## System Architecture

```text
+-----------------------------------------------------------------------------------+
|                                 INGESTION STAGE                                   |
|                                                                                   |
|  SEC EDGAR Submissions API -> Raw HTML/XBRL -> HTML Parser -> Clean Plain Text    |
|  (src/ingest.py)                                               (data/raw/)        |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                                 CHUNKING STAGE                                    |
|                                                                                   |
|  Clean Text -> Paragraph-Aware Sliding Window -> Metadata Enriched JSONL Chunks   |
|  (src/chunk.py) (~650 words, 100 overlap)        (data/processed/chunks.jsonl)    |
+------------------------------------------+----------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------+
|                                 INDEXING STAGE                                    |
|                                                                                   |
|  JSONL Chunks -> SentenceTransformer (bge-small-en-v1.5) -> Local ChromaDB Index  |
|  (src/embed_store.py)                                       (data/chroma_db/)     |
+------------------------------------------+----------------------------------------+
                                           |
                    +----------------------+----------------------+
                    |                                             |
                    v                                             v
+---------------------------------------+     +-------------------------------------+
|            QUERY & RETRIEVAL          |     |          EVALUATION SUITE           |
|                                       |     |                                     |
|  User Query (query.py / retrieve.py)  |     |  Curated Questions                  |
|     |                                 |     |  (eval/eval_questions.json)         |
|     v                                 |     |     |                               |
|  Query Embedding (BGE Query Prefix)   |     |     v                               |
|     |                                 |     |  Retriever + Grounding Benchmark    |
|     v                                 |     |  (eval/run_eval.py)                 |
|  Cosine Similarity Search (ChromaDB)  |     |     |                               |
|     |                                 |     |     v                               |
|     v                                 |     |  Metrics (Hit@K, MRR, Grounding)    |
|  Top-K Retrieved Context              |     |     |                               |
|     |                                 |     |     v                               |
|     v                                 |     |  Scorecard & Benchmark Report       |
|  Groq LLM Generation (generate.py)    |     |  (eval/eval_report.md)              |
|     |                                 |     +-------------------------------------+
|     v                                 |
|  Answer with Bracket Citations [1][2] |
+---------------------------------------+
```

---

## Technical Specifications

| Component | Implementation | Notes |
| :--- | :--- | :--- |
| Ingestion Engine | `httpx`, `BeautifulSoup4` | Fetches directly from SEC EDGAR with rate-limit compliance |
| Chunking Strategy | Paragraph-aware sliding window | ~650 words per chunk, 100-word overlap, preserved metadata |
| Embedding Model | `BAAI/bge-small-en-v1.5` | 384-dimensional dense vectors, normalized for cosine similarity |
| Query Instruction | `Represent this sentence for searching relevant passages: ` | Prepended to search queries for asymmetric retrieval |
| Vector Database | ChromaDB (`PersistentClient`) | Embedded HNSW index, persistent SQLite storage |
| LLM Provider | Groq API (`openai/gpt-oss-120b` / `openai/gpt-oss-20b`) | Temperature 0.1, strict grounded prompt |
| Citation Format | Source bracket mapping (`[1]`, `[2]`, `[1]`) | Regex extraction with link back to chunk metadata |
| Evaluation Framework | Custom automated benchmark | Evaluates Hit Rate @ K, MRR, Context Recall, Grounding |

---

## Directory Structure

```text
.
|-- data/
|   |-- chroma_db/              # Persistent ChromaDB vector index
|   |-- filings_manifest.json   # Metadata manifest for ingested filings
|   |-- processed/
|   |   `-- chunks.jsonl        # Tokenized/chunked dataset with metadata
|   `-- raw/                    # Cleaned plain text filings
|-- eval/
|   |-- eval_questions.json     # 16-question curated benchmark dataset
|   |-- eval_report.md          # Generated evaluation report and scorecard
|   |-- eval_results.json       # Raw evaluation metrics in JSON format
|   `-- run_eval.py             # Evaluation benchmark runner
|-- src/
|   |-- __init__.py
|   |-- chunk.py                # Paragraph-aware sliding window chunker
|   |-- config.py               # Central configuration and environment loader
|   |-- embed_store.py          # Vector embedding and ChromaDB indexing
|   |-- generate.py             # Groq LLM integration and citation formatting
|   |-- ingest.py               # SEC EDGAR scraper and HTML normalizer
|   |-- pipeline.py             # End-to-end query answering pipeline
|   `-- retrieve.py             # Query vector search with optional filtering
|-- .env.example                # Template for environment variables
|-- .gitignore                  # Git exclusion rules
|-- query.py                    # Root CLI entrypoint for interactive queries
|-- README.md                   # System documentation
`-- requirements.txt            # Python package dependencies
```

---

## Setup and Installation

### Prerequisites

- Python 3.10 or higher
- A Groq API key (for answer generation)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Manav-Arora4/RAG-bot-SEC-filings.git
   cd RAG-bot-SEC-filings
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:
   Copy `.env.example` to `.env` and set your credentials:
   ```bash
   cp .env.example .env
   ```
   Edit `.env`:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   SEC_USER_AGENT=FinRAG-Mini user@example.com
   ```

---

## Pipeline Execution

The system can be run step-by-step from raw data ingestion to indexed vector search:

### 1. Ingestion
Downloads 2023 Form 10-K filings from SEC EDGAR for the default ticker list (`AAPL`, `MSFT`, `JPM`, `TSLA`, `WMT`), strips markup, and writes cleaned text files to `data/raw/`.

```bash
python -m src.ingest
```

Optional arguments:
- `--tickers AAPL MSFT NVDA`: Specify custom tickers.
- `--fiscal-year 2023`: Specify fiscal year filter (use `0` for latest available).
- `--manual-dir path/to/files`: Ingest and clean local `.txt` or `.html` files instead of downloading.

### 2. Chunking
Splits cleaned text files into overlapping paragraph blocks with full metadata retention (ticker, company, fiscal year, accession number, source URL, chunk index).

```bash
python -m src.chunk
```

Optional arguments:
- `--chunk-words 650`: Approximate word count per chunk.
- `--overlap-words 100`: Word overlap between consecutive chunks.

### 3. Embedding and Indexing
Generates 384-dimensional embeddings using `BAAI/bge-small-en-v1.5` and persists vectors and metadata into ChromaDB.

```bash
python -m src.embed_store
```

Optional arguments:
- `--rebuild`: Delete existing collection and recreate index from scratch.
- `--batch-size 64`: Batch size for embedding and vector upserts.

### 4. Direct Retrieval
Query the vector store directly to inspect retrieved chunks and similarity scores without calling an LLM:

```bash
python -m src.retrieve "What was Microsoft's revenue for Intelligent Cloud in fiscal 2023?"
```

Optional arguments:
- `--top-k 5`: Number of chunks to retrieve.
- `--ticker MSFT`: Restrict retrieval to a specific company filing.

### 5. Query and Generation
Run the complete RAG pipeline to generate a grounded, cited answer:

```bash
python query.py "What were Apple's total net sales in fiscal year 2023?"
```

Output format:
```text
Apple's total net sales for fiscal year 2023 were $383,285 million [1].

Citations:
- [1] Apple Inc. (AAPL) FY2023 10-K (AAPL_2023_0016) — https://www.sec.gov/Archives/edgar/data/320193/...
```

---

## Evaluation Framework

The project includes an automated benchmarking suite in `eval/` to measure retrieval accuracy, factual grounding, and refusal fidelity.

### Benchmark Dataset

The evaluation dataset (`eval/eval_questions.json`) contains 16 curated test questions categorized into:
- **Factual Financial Metrics**: Specific balance sheet and income statement line items.
- **Qualitative and Strategic Information**: Risk disclosures, supply chain dependencies, AI risks.
- **Cross-Company Comparisons**: Multi-filing synthesis queries.
- **Negative Controls**: Out-of-corpus queries (e.g., non-indexed companies) to verify hallucination refusal.

### Metrics Computed

1. **Retrieval Hit Rate @ K (Hit@1, Hit@3, Hit@K)**: Proportion of queries where at least one ground-truth source document chunk is retrieved in top-k.
2. **Mean Reciprocal Rank (MRR)**: Evaluates the ranking position of the first relevant chunk ($1/\text{rank}$).
3. **Context Keyword Recall**: Verifies that the retrieved chunk text contains required financial terms/values.
4. **Factual Grounding Accuracy**: Verifies that the LLM generated answer matches ground-truth values without fabricating figures.
5. **Citation Validity Rate**: Verifies that all cited bracket numbers (`[1]`, `[2]`) correspond to valid retrieved chunk IDs.
6. **Refusal Fidelity**: Verifies that the model explicitly states lack of context for out-of-corpus queries.

### Running Evaluation

Run retrieval-only benchmark (offline, no LLM API calls):
```bash
python -m eval.run_eval --skip-generation
```

Run full end-to-end evaluation:
```bash
python -m eval.run_eval
```

Outputs:
- Markdown summary report: `eval/eval_report.md`
- Raw results JSON: `eval/eval_results.json`

### Benchmark Baseline Results

| Metric | Score | Target Threshold | Status |
| :--- | :---: | :---: | :---: |
| Retrieval Hit Rate @ 1 | 100.0% | >= 70.0% | PASS |
| Retrieval Hit Rate @ 3 | 100.0% | >= 85.0% | PASS |
| Retrieval Hit Rate @ 5 | 100.0% | >= 90.0% | PASS |
| Mean Reciprocal Rank (MRR) | 1.000 | >= 0.800 | PASS |
| Citation Validity Rate | 100.0% | >= 90.0% | PASS |
| Context Keyword Recall | 81.2% | >= 85.0% | WARN |
| Factual Grounding Accuracy | 81.2% | >= 85.0% | WARN |

---

## Configuration Reference

Configuration defaults are maintained in [`src/config.py`](file:///c:/Users/Totem/Documents/RAG%20mini%20project/src/config.py):

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-small-en-v1.5` | Hugging Face embedding model |
| `CHROMA_COLLECTION_NAME` | `filings` | Chroma collection identifier |
| `BGE_QUERY_PREFIX` | `Represent this sentence for searching relevant passages: ` | Asymmetric retrieval prefix |
| `DEFAULT_EMBED_BATCH_SIZE` | `64` | Embedding computation batch size |
| `DEFAULT_TOP_K` | `5` | Default number of retrieved chunks |
| `DEFAULT_LLM_PROVIDER` | `groq` | Generation provider backend |
| `DEFAULT_GROQ_MODEL` | `openai/gpt-oss-120b` | Default Groq model |
| `DEFAULT_SEC_USER_AGENT` | `finrag-mini/0.1 contact@example.com` | Default fallback user agent header |

