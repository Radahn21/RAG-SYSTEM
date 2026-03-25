# RAG Ingestion Pipeline


A complete end-to-end manual vectorization RAG (Retrieval-Augmented Generation) ingestion pipeline for Azure AI Search.

## Features

- **Azure Blob Storage Integration**: Download PDF and text files from Azure Blob Storage
- **PDF Text Extraction**: Extract text from PDFs with page number tracking using PyPDF2
- **Smart Chunking**: Character-based chunking with overlap and intelligent break points
- **Local Embeddings**: Generate embeddings locally using sentence-transformers (no cloud API calls)
- **Azure AI Search**: Upload documents with vectors to Azure AI Search for retrieval
- **RBAC Authentication**: Uses Entra ID (InteractiveBrowserCredential) - no API keys needed

## Prerequisites

### Azure Resources

1. **Azure AI Search** service with:
   - Index named `rag-index` (or will be created automatically)
   
2. **Azure Blob Storage** account with:
   - Container with PDF/TXT/MD files to ingest

### Required RBAC Roles

Assign these roles to your Entra ID account:

| Service | Role | Purpose |
|---------|------|---------|
| Azure AI Search | **Search Index Data Contributor** | Upload documents to index |
| Azure AI Search | **Search Index Data Reader** | Query the index |
| Azure AI Search | **Search Service Contributor** | Create/manage index schema |
| Azure Storage | **Storage Blob Data Reader** | Read blobs from container |

### Python Environment

- Python 3.10 or higher
- `py` launcher on Windows

## Installation

1. **Clone/download the repository**:
   ```powershell
   cd c:\RAG-Ingestion
   ```

2. **Create and activate virtual environment** (optional but recommended):
   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**:
   ```powershell
   py -m pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```powershell
   copy .env.example .env
   ```
   
   Edit `.env` with your Azure settings:
   ```ini
   AZURE_SEARCH_ENDPOINT=https://your-search-service.search.windows.net
   AZURE_SEARCH_INDEX=rag-index
   AZURE_STORAGE_ACCOUNT_URL=https://yourstorageaccount.blob.core.windows.net
   AZURE_STORAGE_CONTAINER=raw-docs
   ```

## Usage

### Run Ingestion Pipeline

Ingest all documents from Azure Blob Storage:

```powershell
py -m src.ingest
```

A browser window will open for Azure authentication on first run.

**Options:**
```powershell
# Ingest only files with a specific prefix
py -m src.ingest --prefix "folder/subfolder/"

# Force recreate the index (WARNING: deletes existing data)
py -m src.ingest --force-reindex

# Verbose logging
py -m src.ingest --verbose
```

### Query the Index

**Single query:**
```powershell
py -m src.query "What is the document about?"
```

**Hybrid search (vector + keyword):**
```powershell
py -m src.query --hybrid "machine learning models"
```

**More results:**
```powershell
py -m src.query --top 10 "neural networks"
```

**Interactive mode:**
```powershell
py -m src.query
```

**JSON output:**
```powershell
py -m src.query --json "search query" > results.json
```

### Run the Web App

Launch the backend and frontend together from the existing Python environment:

```powershell
py -m src.web.app
```

Then open `http://127.0.0.1:8000` in your browser.

The web UI provides:
- Retrieval-only and RAG modes
- Hybrid or vector-only search
- Top-K control and optional OData filter
- Grounded answer view with citations and verification status
- Retrieved chunk cards for inspection

### Module Testing

Test individual components:

```powershell
# Test authentication
py -m src.auth

# Test blob storage connection
py -m src.blob_reader

# Test embedding model
py -m src.embedder

# Test search connection
py -m src.search_client

# Test index schema
py -m src.schema

# Test PDF extraction (with file)
py -m src.pdf_text "path/to/document.pdf"

# Test chunking (with file)
py -m src.chunker "path/to/document.pdf"
```

## Project Structure

```
RAG-Ingestion/
├── src/
│   ├── __init__.py              # Package marker
│   ├── config.py                # Environment configuration
│   ├── auth.py                  # Azure authentication (Entra ID)
│   ├── blob_reader.py           # Azure Blob Storage operations
│   ├── pdf_text.py              # PDF/text extraction
│   ├── chunker.py               # Text chunking with overlap
│   ├── embedder.py              # Local embeddings (sentence-transformers)
│   ├── search_client.py         # Azure AI Search client
│   ├── schema.py                # Index schema management
│   ├── ingest.py                # Ingestion orchestrator
│   ├── query.py                 # Query interface (retrieval-only & RAG modes)
│   ├── retriever.py             # Retrieval orchestrator (transforms + fusion + post-processing)
│   ├── output_formatter.py      # Grouped output, citations, top findings
│   ├── context_assembler.py     # Context block assembly for LLM prompts
│   ├── retrieval_logger.py      # Debug logging, score distribution, JSON logs
│   ├── prompt_builder.py        # Prompt construction for RAG
│   ├── answer_generator.py      # LLM answer generation orchestrator
│   ├── verify.py                # Citation coverage & grounding checks
│   ├── query_transform/         # Query transformation package
│   │   ├── __init__.py
│   │   ├── router.py            # Query classification & routing
│   │   ├── heuristics.py        # Non-LLM expansion/decomposition
│   │   └── fusion.py            # RRF & score-max merge
│   ├── post_retrieval/          # Post-retrieval processing package
│   │   ├── __init__.py
│   │   ├── dedupe.py            # Chunk deduplication
│   │   ├── diversity.py         # Cross-document diversity
│   │   └── reranker.py          # Optional cross-encoder reranking
│   ├── llm/                     # LLM integration package
│   │   ├── __init__.py
│   │   └── client.py            # Azure AI Projects inference client
│   └── web/                     # FastAPI + browser UI
│       ├── __init__.py
│       ├── app.py               # Web server and JSON API
│       ├── templates/
│       │   └── index.html       # Application shell
│       └── static/
│           ├── styles.css       # UI styling
│           └── app.js           # Browser interaction logic
├── eval/
│   ├── golden_questions.json    # Golden test questions
│   └── run_eval.py              # Evaluation runner
├── logs/                        # Retrieval debug logs (auto-created)
├── requirements.txt
├── .env.example
├── .env
└── README.md
```

## Index Schema

The pipeline uses (or creates) an index with this schema:

| Field | Type | Description |
|-------|------|-------------|
| `id` | String | Unique chunk ID (key) |
| `content` | String | Chunk text content |
| `content_vector` | Collection(Single) | 384-dim embedding vector |
| `source_file` | String | Original filename |
| `blob_path` | String | Full blob path |
| `chunk_id` | Int32 | Chunk number within document |
| `page_info` | String | PDF page range (e.g., "p3-p5") |
| `created_at` | DateTimeOffset | Ingestion timestamp |

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AZURE_SEARCH_ENDPOINT` | ✅ | - | Azure AI Search endpoint URL |
| `AZURE_SEARCH_INDEX` | ❌ | `rag-index` | Target index name |
| `AZURE_STORAGE_ACCOUNT_URL` | ✅ | - | Storage account URL |
| `AZURE_STORAGE_CONTAINER` | ✅ | - | Blob container name |
| `AZURE_STORAGE_PREFIX` | ❌ | `` | Blob prefix filter |
| `LOCAL_EMBED_MODEL` | ❌ | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHUNK_SIZE_CHARS` | ❌ | `1500` | Chunk size in characters |
| `CHUNK_OVERLAP_CHARS` | ❌ | `200` | Overlap between chunks |
| `TOP_K` | ❌ | `5` | Default search results |
| `DOWNLOAD_DIR` | ❌ | `./data/downloads` | Local download directory |

#### Retrieval & LLM Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_QUERY_TRANSFORMS` | `false` | Enable multi-query expansion/decomposition |
| `ENABLE_RERANKER` | `false` | Enable cross-encoder reranking |
| `ENABLE_LLM` | `false` | Enable GPT-5.3 answer generation |
| `LLM_MODEL_DEPLOYMENT_NAME` | `gpt-5.3-chat` | Azure AI model deployment name |
| `MAX_CONTEXT_CHARS` | `12000` | Max context size for LLM prompts |
| `NUM_QUERY_EXPANSIONS` | `3` | Number of query expansion variants |
| `NUM_SUB_QUERIES` | `3` | Max sub-queries for decomposition |
| `TOP_K_RETRIEVE` | `30` | Search depth (before post-processing) |
| `TOP_K_FINAL` | `5` | Final results after post-processing |

### Embedding Model

Default model: `sentence-transformers/all-MiniLM-L6-v2`
- Produces 384-dimensional vectors
- Fast and efficient for local inference
- Good balance of quality and speed

To use a different model, update `LOCAL_EMBED_MODEL` and ensure your index vector dimensions match.
If the embedding dimension changes, recreate the index with `py -m src.ingest --force-reindex`.

## Troubleshooting

### Authentication Errors

**Browser doesn't open:**
- Ensure you're running in an environment that supports browser popups
- Try running from a terminal (not embedded in IDE)

**401 Unauthorized:**
- Token may have expired - run again to re-authenticate
- Verify your Entra ID account has the required RBAC roles

### Storage Errors

**403 Forbidden on blob access:**
- Missing `Storage Blob Data Reader` role
- Check storage account firewall settings
- Verify the container name is correct

**Container not found:**
- Double-check `AZURE_STORAGE_CONTAINER` value
- Ensure the container exists in the storage account

### Search Errors

**403 Forbidden on search:**
- Missing `Search Index Data Contributor` or `Search Index Data Reader` role
- Roles can take a few minutes to propagate after assignment

**Index not found:**
- Run `py -m src.ingest` to create the index
- Or check `AZURE_SEARCH_INDEX` matches your existing index

**Vector dimension mismatch:**
- Your index expects different dimensions than the embedding model produces
- Either recreate the index with `--force-reindex` or use a compatible embedding model

### Performance

**Slow embedding generation:**
- First run downloads the model (~90MB)
- Consider using GPU if available (automatically detected)
- Reduce `CHUNK_SIZE_CHARS` for more but smaller chunks

**Out of memory:**
- Process fewer files at a time using `--prefix`
- Reduce batch size in upload

## Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    INGESTION PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Azure Blob  │ -> │  Download   │ -> │  Extract    │     │
│  │  Storage    │    │   Files     │    │   Text      │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                              │              │
│                                              ▼              │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Azure AI    │ <- │   Upload    │ <- │  Chunk &    │     │
│  │   Search    │    │  Documents  │    │   Embed     │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                 RETRIEVAL PIPELINE (enhanced)                 │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────┐        │
│  │  User    │-->│  Query       │-->│  Multi-Query  │        │
│  │  Query   │   │  Transforms  │   │  Search       │        │
│  └──────────┘   │  (optional)  │   │  (per variant)│        │
│                 └──────────────┘   └───────┬───────┘        │
│                                            │                │
│  ┌──────────┐   ┌──────────────┐   ┌───────▼───────┐        │
│  │ Formatted│<--│  Dedup +     │<--│  RRF Fusion   │        │
│  │ Output   │   │  Diversity + │   │  (merge)      │        │
│  └──────────┘   │  Reranker    │   └───────────────┘        │
│                 └──────────────┘                            │
│                                                              │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│              RAG + LLM MODE (optional, --mode rag)           │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────┐        │
│  │ Retrieved│-->│  Context     │-->│  Prompt       │        │
│  │ Chunks   │   │  Assembly    │   │  Builder      │        │
│  └──────────┘   └──────────────┘   └───────┬───────┘        │
│                                            │                │
│  ┌──────────┐   ┌──────────────┐   ┌───────▼───────┐        │
│  │ Answer + │<--│  Verify      │<--│  GPT-5.3      │        │
│  │ Citations│   │  (grounding) │   │  Completion   │        │
│  └──────────┘   └──────────────┘   └───────────────┘        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

## Retrieval-only vs RAG+LLM Modes

The pipeline supports two query modes:

### Retrieval-only (default)

Returns search results grouped by source document with citations and algorithmically extracted top findings. No LLM calls are made.

```powershell
# Basic retrieval
py -m src.query "What are the safety requirements?"

# With query transforms (multi-query expansion)
# Set ENABLE_QUERY_TRANSFORMS=true in .env
py -m src.query "safety requirements and training procedures"

# With reranking
# Set ENABLE_RERANKER=true in .env
py -m src.query --hybrid "fire hazard management"

# Debug mode: score distribution + JSON log
py -m src.query --debug "offshore safety audit"
```

### RAG + LLM (requires ENABLE_LLM=true)

Retrieves context, builds a grounded prompt, and generates an answer with citations using GPT-5.3 via Azure AI Projects. No agent threads — direct chat completion.

```powershell
# Enable in .env: ENABLE_LLM=true
py -m src.query --mode rag "What are the OSHA requirements for confined spaces?"

# JSON output with answer
py -m src.query --mode rag --json "emergency response procedures"

# Interactive mode (toggle with /mode)
py -m src.query
```

### Feature Flags

All features are opt-in via environment variables:

| Feature | Env Var | Default | Effect |
|---------|---------|---------|--------|
| Query transforms | `ENABLE_QUERY_TRANSFORMS` | `false` | Multi-query expansion + RRF fusion |
| Reranker | `ENABLE_RERANKER` | `false` | Cross-encoder reranking (local model) |
| LLM answers | `ENABLE_LLM` | `false` | GPT-5.3 answer generation |

### Evaluation

Run the golden questions evaluation suite:

```powershell
# Retrieval-only evaluation
py eval/run_eval.py

# RAG evaluation (requires ENABLE_LLM=true)
py eval/run_eval.py --mode rag

# Verbose with hybrid search
py eval/run_eval.py --verbose --hybrid
```

## License

MIT License - feel free to use and modify.
