# RAG System — End-to-End Retrieval-Augmented Generation on Azure

A production-grade, keyless RAG pipeline built on **Azure AI Search**, **Azure Blob Storage**, and **Azure AI Foundry (GPT-5.3)** — fully authenticated through **Microsoft Entra ID** with zero API keys stored anywhere in the codebase.

This project demonstrates a complete document intelligence workflow: ingest PDFs from cloud storage, chunk and embed them locally, index them into Azure AI Search, retrieve relevant context with advanced query processing, and optionally generate grounded answers with citations using GPT-5.3 through Azure AI Foundry's Responses API.

---

## Highlights

| Capability | Detail |
|---|---|
| **Keyless authentication** | Entra ID (`InteractiveBrowserCredential`) across every Azure service — no API keys, no connection strings |
| **Local embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) — no cloud embedding API calls |
| **Hybrid search** | Vector similarity + BM25 keyword search via Azure AI Search |
| **Query transforms** | Synonym expansion, conjunction decomposition, step-back queries — all non-LLM heuristics |
| **Multi-query fusion** | Reciprocal Rank Fusion (RRF) across query variants |
| **Post-retrieval processing** | Content deduplication, cross-document diversity enforcement, optional cross-encoder reranking |
| **Grounded LLM answers** | GPT-5.3 via Azure AI Foundry Responses API with citation extraction and post-generation verification |
| **Web UI** | FastAPI backend + custom browser interface for interactive retrieval and RAG |
| **Evaluation framework** | Golden-question suite with keyword recall metrics (achieved 94% recall, 6/8 perfect) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       INGESTION PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Azure Blob ──> Download ──> Extract Text ──> Chunk & Embed   │
│    Storage         Files      (PDF/TXT/MD)     (local model)   │
│                                                     │          │
│                                                     ▼          │
│                                              Azure AI Search   │
│                                              (upload vectors)  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL PIPELINE (enhanced)                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   User Query ──> Query Transforms ──> Multi-Query Search        │
│                  (optional)            (per variant)             │
│                                             │                   │
│                                             ▼                   │
│   Formatted  <── Dedup + Diversity  <── RRF Fusion              │
│    Output        + Reranker (opt.)      (merge scores)          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              RAG + LLM MODE (optional, --mode rag)              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Retrieved  ──> Context     ──> Prompt    ──> GPT-5.3          │
│    Chunks        Assembly        Builder       (Foundry)        │
│                                                    │            │
│   Answer +   <── Verify      <── Parse     <──────┘            │
│   Citations      (grounding)     Citations                      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## How Keyless Authentication Works

This project uses **zero API keys** across all Azure services. Every call — blob reads, search queries, index management, and LLM inference — is authenticated through Microsoft Entra ID tokens.

The authentication chain:

1. **`auth.py`** creates an `InteractiveBrowserCredential` (with `DeviceCodeCredential` fallback) that opens a browser for sign-in.
2. The resulting credential is passed to every Azure SDK client: `BlobServiceClient`, `SearchClient`, `SearchIndexClient`, and `AIProjectClient`.
3. For LLM inference, **`llm/client.py`** creates an `AIProjectClient` with the Entra credential, then calls `project_client.get_openai_client()` which returns an OpenAI-compatible client pre-configured with the Foundry endpoint and token — no API key needed.
4. The LLM client then calls `client.responses.create(...)` (the Responses API), which is the working inference path in this Foundry environment. The standard `chat.completions` endpoint is disabled by policy.

**Key insight:** `client.py` does not call OpenAI directly — it calls Foundry, and Foundry calls the model on your behalf using your Entra identity token.

---

## Required RBAC Roles

Assign these roles to your Entra ID account on the corresponding Azure resources:

### Azure Blob Storage

| Role | Purpose |
|---|---|
| **Storage Blob Data Reader** | List and download source documents from containers |

### Azure AI Search

| Role | Purpose |
|---|---|
| **Search Index Data Reader** | Query the index during retrieval |
| **Search Index Data Contributor** | Upload chunk documents and vectors during ingestion |
| **Search Service Contributor** | Create or modify the index schema (only needed if the pipeline manages the index lifecycle) |

### Azure AI Foundry / AI Project

| Role | Purpose |
|---|---|
| **Azure AI Developer** | Full access to the Foundry project: model inference via `get_openai_client()`, agent operations, project-level resources |

> **Note:** Roles can take a few minutes to propagate after assignment. If you get 403 errors immediately after assigning a role, wait and retry.

### Minimum by Scenario

| Scenario | Roles needed |
|---|---|
| Ingestion + retrieval only | Storage Blob Data Reader, Search Index Data Reader, Search Index Data Contributor, Search Service Contributor |
| Retrieval + GPT-5.3 answers | Search Index Data Reader, Azure AI Developer |
| Full pipeline (all features) | All five roles above |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Authentication | `azure-identity` — InteractiveBrowserCredential, DeviceCodeCredential |
| Blob Storage | `azure-storage-blob` — BlobServiceClient with RBAC |
| Search | `azure-search-documents` — SearchClient, SearchIndexClient with RBAC |
| LLM Inference | `azure-ai-projects` — AIProjectClient → get_openai_client() → Responses API |
| Embeddings | `sentence-transformers` — all-MiniLM-L6-v2 (384-dim, local) |
| Reranking | `sentence-transformers` — cross-encoder/ms-marco-MiniLM-L-6-v2 (optional, local) |
| Web backend | FastAPI + Uvicorn |
| Web frontend | Vanilla HTML/CSS/JS (Jinja2 templates) |
| PDF processing | PyPDF2 |
| Config | python-dotenv |

---

## Project Structure

```
├── auth.py                  # Entra ID credential (keyless auth layer)
├── config.py                # Environment configuration loader
├── blob_reader.py           # Azure Blob Storage download operations
├── pdf_text.py              # PDF and text file extraction
├── chunker.py               # Character-based chunking with overlap
├── embedder.py              # Local embedding generation (sentence-transformers)
├── search_client.py         # Azure AI Search client (RBAC)
├── schema.py                # Index schema creation and validation
├── ingest.py                # Ingestion orchestrator (blob → chunk → embed → index)
├── query.py                 # CLI query interface (retrieval-only & RAG modes)
├── retriever.py             # Retrieval orchestrator (transforms → search → fusion → post-processing)
├── output_formatter.py      # Grouped output, citations, algorithmic top findings
├── context_assembler.py     # Context block assembly for LLM prompts
├── retrieval_logger.py      # JSON debug logs, score distribution, variant maps
├── prompt_builder.py        # System/user prompt construction for RAG
├── answer_generator.py      # LLM answer generation with verification
├── verify.py                # Citation coverage and grounding checks
├── query_transform/
│   ├── router.py            # Query classification (simple/complex/ambiguous)
│   ├── heuristics.py        # Non-LLM rewrite, expand, decompose, step-back
│   └── fusion.py            # Reciprocal Rank Fusion (RRF) and score-max merge
├── post_retrieval/
│   ├── dedupe.py            # Two-pass deduplication (key + content overlap)
│   ├── diversity.py         # Cross-document diversity (per-source caps)
│   └── reranker.py          # Optional cross-encoder reranking
├── llm/
│   └── client.py            # Azure AI Projects inference client (Responses API)
├── web/
│   ├── app.py               # FastAPI web server and JSON API
│   ├── templates/
│   │   └── index.html       # Browser application shell
│   └── static/
│       ├── styles.css        # UI styling
│       └── app.js            # Frontend interaction logic
├── eval/
│   ├── golden_questions.json # 8 golden test questions
│   └── run_eval.py           # Evaluation runner with keyword recall metrics
├── _archived_agent_test.py   # Reference: Foundry agent API connectivity test
├── requirements.txt
├── .env.example
└── README.md
```

---

## Installation

```powershell
# Clone the repo
git clone https://github.com/t-hamada_microsoft/Testing-RAG-System.git
cd Testing-RAG-System

# Create virtual environment
py -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
py -m pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env with your Azure endpoints and settings
```

---

## Usage

### Ingestion

```powershell
# Ingest all documents from Azure Blob Storage
py -m src.ingest

# Ingest a subfolder only
py -m src.ingest --prefix "folder/subfolder/"

# Force-recreate the index (deletes existing data)
py -m src.ingest --force-reindex

# Verbose logging
py -m src.ingest --verbose
```

### Query — Retrieval Only (default)

```powershell
# Basic vector search
py -m src.query "What are the safety requirements?"

# Hybrid search (vector + keyword)
py -m src.query --hybrid "machine learning models"

# More results
py -m src.query --top 10 "neural networks"

# Debug mode (score distribution + JSON log)
py -m src.query --debug "safety audit"

# JSON output
py -m src.query --json "search query" > results.json

# Interactive mode
py -m src.query
```

### Query — RAG Mode (GPT-5.3 answers)

Requires `ENABLE_LLM=true` in `.env`.

```powershell
# Single query with grounded answer
py -m src.query --mode rag "What are the OSHA requirements for confined spaces?"

# JSON output with answer + citations
py -m src.query --mode rag --json "emergency response procedures"

# Interactive mode (toggle with /mode command)
py -m src.query
```

### Web App

```powershell
py -m src.web.app
# Open http://127.0.0.1:8000
```

The web UI provides:
- **Retrieval-only and RAG modes** with a single toggle
- **Hybrid or vector-only** search selection
- **Top-K and OData filter** controls
- **Grounded answer panel** with citations and verification status
- **Retrieved chunk cards** for evidence inspection
- **Preset queries** for quick testing

### Evaluation

```powershell
# Run retrieval-only evaluation
py eval/run_eval.py

# RAG evaluation
py eval/run_eval.py --mode rag

# Verbose with hybrid search
py eval/run_eval.py --verbose --hybrid
```

### Module Testing

Each module can be tested independently:

```powershell
py -m src.auth             # Test Entra ID authentication
py -m src.blob_reader      # Test blob storage connection
py -m src.embedder         # Test embedding model
py -m src.search_client    # Test search connection
py -m src.schema           # Test index schema
py -m src.pdf_text "file"  # Test PDF extraction
py -m src.chunker "file"   # Test chunking
```

---

## Index Schema

| Field | Type | Description |
|---|---|---|
| `id` | String | Unique chunk ID (document key) |
| `content` | String | Chunk text content |
| `content_vector` | Collection(Single) | 384-dim embedding vector |
| `source_file` | String | Original filename |
| `blob_path` | String | Full blob storage path |
| `chunk_id` | Int32 | Chunk number within document |
| `page_info` | String | PDF page range (e.g., "p3-p5") |
| `created_at` | DateTimeOffset | Ingestion timestamp |

---

## Configuration

### Core Settings

| Variable | Required | Default | Description |
|---|---|---|---|
| `AZURE_SEARCH_ENDPOINT` | Yes | — | Azure AI Search endpoint URL |
| `AZURE_SEARCH_INDEX` | No | `rag-index` | Target index name |
| `AZURE_STORAGE_ACCOUNT_URL` | Yes | — | Storage account URL |
| `AZURE_STORAGE_CONTAINER` | Yes | — | Blob container name |
| `AZURE_STORAGE_PREFIX` | No | ` ` | Blob prefix filter |
| `AZURE_AI_PROJECT_ENDPOINT` | Yes (for LLM) | — | Azure AI Foundry project endpoint |
| `LOCAL_EMBED_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHUNK_SIZE_CHARS` | No | `1500` | Chunk size in characters |
| `CHUNK_OVERLAP_CHARS` | No | `200` | Overlap between chunks |
| `TOP_K` | No | `5` | Default search results |

### Feature Flags

| Variable | Default | Description |
|---|---|---|
| `ENABLE_QUERY_TRANSFORMS` | `false` | Multi-query expansion + RRF fusion |
| `ENABLE_RERANKER` | `false` | Cross-encoder reranking (local model) |
| `ENABLE_LLM` | `false` | GPT-5.3 answer generation via Foundry |
| `LLM_MODEL_DEPLOYMENT_NAME` | `gpt-5.3-chat` | Model deployment name in Foundry |
| `MAX_CONTEXT_CHARS` | `12000` | Max context window for LLM prompts |
| `TOP_K_RETRIEVE` | `30` | Search depth before post-processing |
| `TOP_K_FINAL` | `5` | Final results after post-processing |

---

## Evaluation Results

Tested against 8 golden questions on an oil & gas safety document corpus:

| Metric | Result |
|---|---|
| **Overall keyword recall** | 94% |
| **Perfect recall (all keywords found)** | 6 out of 8 questions |
| **Search mode** | Vector (default) |
| **LLM used during evaluation** | None (retrieval-only) |

---

## Key Design Decisions

1. **Local embeddings over cloud APIs** — `all-MiniLM-L6-v2` runs locally so ingestion does not depend on external embedding services. This reduces latency, avoids rate limits, and keeps embedding costs at zero.

2. **Responses API over chat.completions** — In this Foundry environment, the standard `chat.completions.create()` endpoint returns `400: API operation not supported for token authentication`. The Responses API (`responses.create()`) is the working path and is used throughout.

3. **Non-LLM query transforms** — Synonym expansion, step-back queries, and conjunction decomposition are all heuristic-based. This keeps retrieval improvements independent of LLM availability and avoids latency overhead.

4. **Post-generation verification** — After GPT generates an answer, `verify.py` checks citation coverage and grounding (word overlap between claims and source context). This catches hallucinated citations and ungrounded claims without a second LLM call.

5. **Feature flags over hardcoded behavior** — Query transforms, reranking, and LLM answers are all opt-in via environment variables. The pipeline works as a pure retrieval system with everything disabled.

---

## Troubleshooting

### Authentication
- **Browser doesn't open:** Run from a standalone terminal, not an embedded IDE terminal
- **401 Unauthorized:** Token expired — re-run to re-authenticate; verify RBAC roles are assigned
- **Tenant mismatch:** Set `AZURE_USE_EXPLICIT_TENANT=true` and verify `AZURE_TENANT_ID`

### Storage
- **403 Forbidden:** Missing `Storage Blob Data Reader` role; check storage firewall settings
- **Container not found:** Verify `AZURE_STORAGE_CONTAINER` value

### Search
- **403 Forbidden:** Missing `Search Index Data Contributor` or `Search Index Data Reader` role
- **Vector dimension mismatch:** Recreate the index with `--force-reindex` or use a compatible embedding model

### LLM / Foundry
- **400 "API operation not supported for token authentication":** This means `chat.completions` is blocked — the codebase already uses the Responses API to avoid this
- **400 "Unsupported parameter: 'temperature'":** Some Foundry model deployments do not support temperature — this has been removed from the request
- **NoneType is not subscriptable on response parsing:** The response output structure varies by model — the codebase walks the output tree safely

---

## License

MIT License — feel free to use and modify.
