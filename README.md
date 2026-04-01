# RAG System — End-to-End Retrieval-Augmented Generation on Azure

A production-grade, keyless RAG pipeline built on **Azure AI Search**, **Azure Blob Storage**, and **Azure AI Foundry (GPT-5.3)** — fully authenticated through **Microsoft Entra ID** with zero API keys stored anywhere in the codebase.

This project demonstrates a complete document intelligence workflow: ingest PDFs from cloud storage, chunk and embed them locally, index them into Azure AI Search, retrieve relevant context with advanced query processing, and optionally generate grounded answers with citations using GPT-5.3 through Azure AI Foundry's Responses API.

---

## Table of Contents

- [Highlights](#highlights)
- [Architecture](#architecture)
- [How Keyless Authentication Works](#how-keyless-authentication-works)
- [Required RBAC Roles](#required-rbac-roles)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Ingestion Pipeline — Deep Dive](#ingestion-pipeline--deep-dive)
  - [Step 1: Index Creation](#step-1-index-creation--schema-validation)
  - [Step 2: Blob Download](#step-2-blob-download)
  - [Step 3: Text Extraction](#step-3-text-extraction)
  - [Step 4: Chunking](#step-4-chunking)
  - [Step 5: Embedding](#step-5-embedding)
  - [Step 6: Upload](#step-6-upload-to-azure-ai-search)
- [Retrieval Pipeline — Deep Dive](#retrieval-pipeline--deep-dive)
- [RAG + LLM Mode — Deep Dive](#rag--llm-mode--deep-dive)
- [Web UI](#web-ui)
- [Evaluation](#evaluation)
- [Configuration](#configuration)
- [Key Design Decisions](#key-design-decisions)
- [Troubleshooting](#troubleshooting)

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
│                                                                 │
│   blob_reader.py        pdf_text.py        chunker.py           │
│   ┌──────────┐         ┌──────────┐       ┌──────────┐         │
│   │  Azure   │ ──────> │ Extract  │ ────> │  Chunk   │         │
│   │  Blob    │  list & │  Text    │ text  │  with    │         │
│   │  Storage │  d/load │ (PyPDF2) │       │  Overlap │         │
│   └──────────┘         └──────────┘       └────┬─────┘         │
│                                                │               │
│   search_client.py      schema.py         embedder.py          │
│   ┌──────────┐         ┌──────────┐       ┌────▼─────┐         │
│   │  Upload  │ <────── │  Index   │       │  Embed   │         │
│   │  Batches │  500/   │  Schema  │       │  Locally │         │
│   │  to AI   │  batch  │  (HNSW)  │       │ MiniLM   │         │
│   │  Search  │         └──────────┘       └──────────┘         │
│   └──────────┘                                                 │
│                                                                 │
│   Orchestrator: ingest.py  →  run_ingestion()                   │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    RETRIEVAL PIPELINE                            │
│                                                                 │
│   query_transform/          search_client.py     fusion.py      │
│   ┌──────────────┐         ┌──────────────┐    ┌──────────┐    │
│   │ Heuristics:  │ ──────> │ Multi-Query  │ ─> │   RRF    │    │
│   │ rewrite,     │ N       │ Search       │    │  Fusion  │    │
│   │ expand,      │ queries │ (per variant)│    │  (k=60)  │    │
│   │ decompose,   │         └──────────────┘    └────┬─────┘    │
│   │ step_back    │                                  │          │
│   └──────────────┘                                  ▼          │
│                                                                 │
│   output_formatter.py      post_retrieval/     retriever.py     │
│   ┌──────────────┐        ┌──────────────┐    ┌──────────┐     │
│   │  Group by    │ <───── │ Dedup →      │ <─ │ Orchestr.│     │
│   │  Source,     │        │ Diversity →  │    │ retrieve()     │
│   │  Citations,  │        │ Rerank (opt) │    └──────────┘     │
│   │  Top Findings│        └──────────────┘                      │
│   └──────────────┘                                              │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│              RAG + LLM MODE (optional, --mode rag)              │
│                                                                 │
│   context_assembler.py    prompt_builder.py    llm/client.py    │
│   ┌──────────────┐       ┌──────────────┐    ┌──────────┐      │
│   │  Assemble    │ ────> │  Build       │ ─> │ GPT-5.3  │      │
│   │  Context     │       │  System +    │    │ Responses│      │
│   │  Blocks      │       │  User Prompt │    │  API     │      │
│   └──────────────┘       └──────────────┘    └────┬─────┘      │
│                                                   │            │
│   answer_generator.py     verify.py          prompt_builder.py  │
│   ┌──────────────┐       ┌──────────────┐    ┌────▼─────┐      │
│   │  Return      │ <──── │  Citation    │ <─ │  Parse   │      │
│   │  Answer +    │       │  Coverage +  │    │ Citations│      │
│   │  Citations + │       │  Grounding   │    │  from    │      │
│   │  Verification│       │  Check       │    │  Answer  │      │
│   └──────────────┘       └──────────────┘    └──────────┘      │
│                                                                 │
│   Orchestrator: answer_generator.py  →  generate_answer()       │
└─────────────────────────────────────────────────────────────────┘
```

---

## How Keyless Authentication Works

This project uses **zero API keys** across all Azure services. Every call — blob reads, search queries, index management, and LLM inference — is authenticated through Microsoft Entra ID tokens.

### The authentication chain

**File:** `auth.py` — function `get_credential()`

1. Creates an `InteractiveBrowserCredential` with `redirect_uri="http://localhost:8400"` and `additionally_allowed_tenants=["*"]`.
2. Creates a `DeviceCodeCredential` as fallback (for headless environments).
3. Wraps both in a `ChainedTokenCredential` — tries browser first, then device code.
4. Stores the result in a module-level singleton `_credential` so it is only created once per process.

**Where the credential is used:**

| Module | Azure Client | What it authenticates |
|---|---|---|
| `blob_reader.py` | `BlobServiceClient(account_url, credential)` | Blob listing and downloads |
| `search_client.py` | `SearchClient(endpoint, index_name, credential)` | Document queries and uploads |
| `search_client.py` | `SearchIndexClient(endpoint, credential)` | Index creation and management |
| `llm/client.py` | `AIProjectClient(endpoint, credential)` → `.get_openai_client()` | GPT-5.3 model inference |
| `_archived_agent_test.py` | `AIProjectClient(endpoint, credential)` → `.agents.create_version()` | Foundry agent operations |

**For LLM inference specifically:**

`llm/client.py` calls `project_client.get_openai_client()` which returns a standard `openai.OpenAI` instance that is automatically configured with the Foundry endpoint URL and a bearer token from your Entra credential. No API key is passed. The client then calls `client.responses.create(...)`:

```
get_credential()           →  Entra ID token
AIProjectClient(cred)      →  Foundry project handle
.get_openai_client()       →  openai.OpenAI pointed at Foundry
.responses.create(...)     →  GPT-5.3 inference (keyless)
```

**Key insight:** `client.py` does not call OpenAI directly — it calls Foundry, and Foundry calls the model on your behalf using your Entra identity token.

### Tenant handling

- By default, `InteractiveBrowserCredential` uses whichever tenant you select at sign-in.
- If `AZURE_TENANT_ID` is set AND `AZURE_USE_EXPLICIT_TENANT=true`, the credential forces that tenant.
- If `AZURE_TENANT_ID` is set but `AZURE_USE_EXPLICIT_TENANT` is not, the tenant ID is logged but not enforced. This avoids local dev failures from a stale or incorrect tenant.

---

## Required RBAC Roles

### Azure Blob Storage

| Role | Purpose | Used by |
|---|---|---|
| **Storage Blob Data Reader** | List and download source documents from containers | `blob_reader.py` |

### Azure AI Search

| Role | Purpose | Used by |
|---|---|---|
| **Search Index Data Reader** | Query the index during retrieval | `search_client.py` → `vector_search()`, `hybrid_search()` |
| **Search Index Data Contributor** | Upload chunk documents and vectors during ingestion | `search_client.py` → `upload_documents()` |
| **Search Service Contributor** | Create or modify the index schema | `schema.py` → `create_index()` |

### Azure AI Foundry / AI Project

| Role | Purpose | Used by |
|---|---|---|
| **Azure AI Developer** | Model inference via `get_openai_client()`, agent operations | `llm/client.py`, `_archived_agent_test.py` |

> **Note:** Roles can take a few minutes to propagate after assignment. If you get 403 errors immediately after assigning a role, wait and retry.

### Minimum by Scenario

| Scenario | Roles needed |
|---|---|
| Ingestion + retrieval only | Storage Blob Data Reader, Search Index Data Reader, Search Index Data Contributor, Search Service Contributor |
| Retrieval + GPT-5.3 answers | Search Index Data Reader, Azure AI Developer |
| Full pipeline (all features) | All five roles above |

---

## Tech Stack

| Layer | Technology | Version used |
|---|---|---|
| Language | Python | 3.14 |
| Authentication | `azure-identity` — InteractiveBrowserCredential, DeviceCodeCredential | ≥1.15.0 |
| Blob Storage | `azure-storage-blob` — BlobServiceClient with RBAC | ≥12.19.0 |
| Search | `azure-search-documents` — SearchClient, SearchIndexClient with RBAC | ≥11.4.0 |
| LLM Inference | `azure-ai-projects` — AIProjectClient → get_openai_client() → Responses API | ≥2.0.0 |
| Embeddings | `sentence-transformers` — all-MiniLM-L6-v2 (384-dim, local) | ≥2.2.0 |
| Reranking | `sentence-transformers` — cross-encoder/ms-marco-MiniLM-L-6-v2 (optional, local) | ≥2.2.0 |
| Deep learning | PyTorch (auto-detects GPU) | ≥2.0.0 |
| Web backend | FastAPI + Uvicorn | ≥0.115.0 |
| Web frontend | Vanilla HTML/CSS/JS (Jinja2 templates) | — |
| PDF processing | PyPDF2 | ≥3.0.0 |
| Config | python-dotenv | ≥1.0.0 |

---

## Project Structure

```
├── auth.py                  # Entra ID credential (keyless auth layer)
├── config.py                # Environment configuration loader (.env → Config dataclass)
├── blob_reader.py           # Azure Blob Storage download operations
├── pdf_text.py              # PDF and text file extraction with page tracking
├── chunker.py               # Character-based chunking with overlap and smart breaks
├── embedder.py              # Local embedding generation (sentence-transformers)
├── search_client.py         # Azure AI Search client (RBAC, vector/hybrid search, upload)
├── schema.py                # Index schema creation, validation, HNSW config
├── ingest.py                # Ingestion orchestrator (blob → extract → chunk → embed → upload)
├── query.py                 # CLI query interface (retrieval-only & RAG modes, interactive)
├── retriever.py             # Retrieval orchestrator (transforms → search → fusion → post-process)
├── output_formatter.py      # Grouped output, citations, algorithmic top findings
├── context_assembler.py     # Context block assembly for LLM prompts (max chars cap)
├── retrieval_logger.py      # JSON debug logs, score distribution, variant maps
├── prompt_builder.py        # System/user prompt construction for RAG
├── answer_generator.py      # LLM answer generation with post-generation verification
├── verify.py                # Citation coverage and grounding checks (word overlap)
├── query_transform/
│   ├── router.py            # Query classification (simple/complex/ambiguous)
│   ├── heuristics.py        # Non-LLM: rewrite, expand (synonyms/acronyms), decompose, step-back
│   └── fusion.py            # Reciprocal Rank Fusion (RRF, k=60) and score-max merge
├── post_retrieval/
│   ├── dedupe.py            # Two-pass deduplication (key match + content overlap ≥80%)
│   ├── diversity.py         # Cross-document diversity (per-source caps, round-robin)
│   └── reranker.py          # Optional cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
├── llm/
│   └── client.py            # Azure AI Projects inference client (Responses API, keyless)
├── web/
│   ├── app.py               # FastAPI web server and JSON API
│   ├── templates/
│   │   └── index.html       # Browser application shell (Jinja2)
│   └── static/
│       ├── styles.css        # UI styling (glassmorphism, responsive)
│       └── app.js            # Frontend interaction logic (fetch → render)
├── eval/
│   ├── golden_questions.json # 8 golden test questions with expected keywords
│   └── run_eval.py           # Evaluation runner with keyword recall metrics
├── _archived_agent_test.py   # Reference: Foundry agent API connectivity test
├── requirements.txt
├── .env.example
└── README.md
```

---

## Installation

```powershell
git clone https://github.com/t-hamada_microsoft/Testing-RAG-System.git
cd Testing-RAG-System

py -m venv .venv
.\.venv\Scripts\Activate.ps1

py -m pip install -r requirements.txt

copy .env.example .env
# Edit .env with your Azure endpoints and settings
```

---

## Ingestion Pipeline — Deep Dive

The ingestion pipeline is orchestrated by `ingest.py` → `run_ingestion()`. It runs five sequential steps. Below is a detailed walkthrough of every function, data structure, algorithm, and parameter involved.

### Entry point

```
py -m src.ingest [--prefix <blob_prefix>] [--force-reindex] [--batch-size 500] [--verbose]
```

The `main()` function in `ingest.py` parses CLI arguments and calls `run_ingestion(prefix, force_reindex, batch_size)`, which returns a summary dict with counts and timing.

---

### Step 1: Index Creation / Schema Validation

**File:** `schema.py`
**Functions:** `index_exists()`, `ensure_index_exists()`, `create_index()`, `validate_index_schema()`, `get_index_schema()`

Before any documents are processed, the pipeline checks whether the target Azure AI Search index exists and has the correct schema.

**Decision tree:**

| Condition | Action |
|---|---|
| `--force-reindex` flag | Delete existing index, recreate from scratch |
| Index does not exist | Create new index via `ensure_index_exists()` |
| Index exists | Validate schema via `validate_index_schema()` |

**Schema definition** — `get_index_schema()` returns a `SearchIndex` object with these fields:

| Field | SDK Type | Properties |
|---|---|---|
| `id` | `SearchFieldDataType.String` | key=True, filterable=True |
| `content` | `SearchFieldDataType.String` | searchable=True (enables BM25 keyword search) |
| `content_vector` | `Collection(Single)` | searchable=True, 384 dimensions, profile="default-vector-profile" |
| `source_file` | `SearchFieldDataType.String` | filterable=True, facetable=True |
| `blob_path` | `SearchFieldDataType.String` | filterable=True |
| `chunk_id` | `SearchFieldDataType.Int32` | filterable=True, sortable=True |
| `page_info` | `SearchFieldDataType.String` | filterable=True |
| `created_at` | `SearchFieldDataType.DateTimeOffset` | filterable=True, sortable=True |

**Vector search configuration** — HNSW (Hierarchical Navigable Small World):

| Parameter | Value | Meaning |
|---|---|---|
| `m` | 4 | Number of bi-directional links per node. Lower = less memory, slightly less recall |
| `efConstruction` | 400 | Candidate list size during index building. Higher = more accurate graph |
| `efSearch` | 500 | Candidate list size during search. Higher = better recall, slower queries |
| `metric` | `cosine` | Distance function. Cosine similarity matches how all-MiniLM-L6-v2 was trained |

**Validation** — `validate_index_schema()` checks:
1. Required fields (`id`, `content`, `content_vector`) exist.
2. Vector dimensions match the loaded embedding model (384 for MiniLM).
3. Raises `ValueError` with an actionable message if validation fails.

---

### Step 2: Blob Download

**File:** `blob_reader.py`
**Functions:** `download_all_blobs()`, `download_blob()`, `list_blobs()`, `get_container_client()`, `get_blob_service_client()`
**Data class:** `BlobInfo(name, local_path, blob_path, size_bytes, content_type, extension)`

**What happens:**

1. `get_blob_service_client()` creates a `BlobServiceClient` authenticated with the Entra credential. No account key.
2. `get_container_client()` returns a `ContainerClient` for the container named in `AZURE_STORAGE_CONTAINER`.
3. `list_blobs(prefix, extensions)` iterates over all blobs in the container (optionally filtered by prefix), yielding only those with extensions in `SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}`.
4. `download_all_blobs()` iterates over each blob, downloads it to `config.download_dir` (default `./data/downloads`), preserving the directory structure from blob storage. Returns a list of `BlobInfo` objects.
5. Each blob is downloaded by `download_blob()`, which calls `blob_client.download_blob().readall()` and writes the bytes to disk.

**BlobInfo fields:**

| Field | Type | Example |
|---|---|---|
| `name` | str | `"safety-docs/procedures.pdf"` |
| `local_path` | Path | `./data/downloads/safety-docs/procedures.pdf` |
| `blob_path` | str | `"raw-docs/safety-docs/procedures.pdf"` |
| `size_bytes` | int | `245760` |
| `content_type` | str or None | `"application/pdf"` |
| `extension` | str | `".pdf"` |

**Error handling:** If a blob fails to download, the error is logged and the pipeline continues with the remaining files. A 403 error includes a message about the missing `Storage Blob Data Reader` role.

---

### Step 3: Text Extraction

**File:** `pdf_text.py`
**Functions:** `extract_text()`, `extract_pdf_text()`, `extract_text_file()`, `get_page_range_for_position()`
**Data class:** `ExtractedText(text, source_file, file_path, file_type, page_texts, total_pages, extraction_errors)`

**Orchestrator call:** `ingest.py` → `extract_documents(blob_infos)` loops over each `BlobInfo` and calls `extract_text(blob_info.local_path)`.

**For PDF files** — `extract_pdf_text()`:

1. Opens the file with `PyPDF2.PdfReader(file_path)`.
2. Iterates over `reader.pages` (0-indexed internally, but stored as 1-indexed page numbers).
3. For each page, calls `page.extract_text()` which uses PyPDF2's built-in text extraction (handles most text-based PDFs; image-based PDFs with no embedded text will produce empty pages).
4. Non-empty page texts are stored as `(page_num, page_text)` tuples in `page_texts`.
5. All page texts are joined with `"\n\n"` (double newline) to form the full document text.
6. Extraction errors per page are captured but do not halt the process.

**For TXT / Markdown files** — `extract_text_file()`:

1. Tries multiple encodings in order: `utf-8`, `utf-8-sig`, `latin-1`, `cp1252`.
2. Uses the first encoding that succeeds.
3. Stores the entire content as a single "page" in `page_texts`.

**Page tracking** — `get_page_range_for_position(extracted, start_char, end_char)`:

This function is called later during chunking to determine which PDF pages a chunk spans. It walks through `page_texts`, tracking cumulative character positions (accounting for the `\n\n` separators between pages), and returns a string like `"p3"` (single page) or `"p3-p5"` (multi-page). For non-PDF files, it returns `""`.

**ExtractedText fields:**

| Field | Type | Description |
|---|---|---|
| `text` | str | Full document text, all pages joined with `\n\n` |
| `source_file` | str | Original filename, e.g. `"procedures.pdf"` |
| `file_path` | Path | Absolute local path to the file |
| `file_type` | str | `"pdf"`, `"txt"`, or `"md"` |
| `page_texts` | List[(int, str)] | Per-page text with 1-indexed page numbers |
| `total_pages` | int | Number of pages (for PDFs) or 1 (for text files) |
| `extraction_errors` | List[str] | Any per-page extraction errors |

---

### Step 4: Chunking

**File:** `chunker.py`
**Functions:** `chunk_text()`, `find_best_break_point()`, `generate_chunk_id()`, `sanitize_id()`
**Data class:** `ChunkRecord(id, content, source_file, blob_path, chunk_id, page_info, char_start, char_end)`

**Orchestrator call:** `ingest.py` → `chunk_documents(documents)` loops over each `(ExtractedText, blob_path)` tuple and calls `chunk_text(extracted, blob_path)`.

**Algorithm — sliding window with smart breaks:**

```
Parameters:
  chunk_size    = CHUNK_SIZE_CHARS   (default 1500 characters)
  chunk_overlap = CHUNK_OVERLAP_CHARS (default 200 characters)
  smart_breaks  = True (always on)
```

1. Start at `current_pos = 0`.
2. Calculate `end_pos = current_pos + chunk_size`.
3. If `smart_breaks` is enabled and `end_pos` is not at the end of the document, call `find_best_break_point(text, end_pos)` to adjust the break position.
4. Extract `text[current_pos:end_pos]`, strip whitespace.
5. Look up `page_info` by calling `get_page_range_for_position(extracted, current_pos, end_pos)`.
6. Generate a deterministic chunk ID with `generate_chunk_id(source_file, chunk_num)`.
7. Create a `ChunkRecord` with all metadata.
8. Calculate the next start position: `next_pos = end_pos - chunk_overlap`. If this would not make progress, use `current_pos + chunk_size // 2`.
9. Repeat until the entire document is consumed.

**Smart break algorithm** — `find_best_break_point(text, target_pos, search_range=100)`:

Searches within ±100 characters of the target position for the best natural boundary, in priority order:

| Priority | Pattern | What it matches |
|---|---|---|
| 1 | `\n\n` | Paragraph break |
| 2 | `[.!?]\s+` | Sentence ending |
| 3 | `[;:]\s+` | Clause break |
| 4 | `,\s+` | Comma |
| 5 | `\s+` | Any whitespace |

It prefers the match closest to (but not past) the target position. If no match is found before the target, it takes the first match after. This ensures chunks end at natural text boundaries rather than mid-word or mid-sentence.

**ID generation** — `generate_chunk_id(source_file, chunk_num)`:

1. `sanitize_id(source_file)` replaces `/`, `\`, spaces, and dots with underscores, then strips all non-alphanumeric characters (except `-`, `_`, `=`). Falls back to an MD5 hash if the sanitized result is empty. Truncates to 200 characters.
2. Returns `"{sanitized_name}__chunk_{chunk_num:04d}"` — e.g. `"procedures_pdf__chunk_0003"`.

**Why 1500 chars with 200 overlap?**

- 1500 characters is roughly 250–375 English words. This fits well within the 512-token input window of all-MiniLM-L6-v2 (which was trained on passages of this length).
- 200 characters of overlap ensures that sentences at the boundary of one chunk also appear in the next, so no information is lost at chunk edges.
- If you change these values, the existing index must be re-ingested because chunk boundaries and IDs will change.

**ChunkRecord fields:**

| Field | Type | Example |
|---|---|---|
| `id` | str | `"procedures_pdf__chunk_0003"` |
| `content` | str | The chunk text (up to ~1500 chars) |
| `source_file` | str | `"procedures.pdf"` |
| `blob_path` | str | `"raw-docs/procedures.pdf"` |
| `chunk_id` | int | `3` (0-indexed) |
| `page_info` | str | `"p5-p6"` |
| `char_start` | int | Character start position in original text |
| `char_end` | int | Character end position in original text |

---

### Step 5: Embedding

**File:** `embedder.py`
**Functions:** `embed_chunks()`, `embed_texts()`, `embed_text()`, `get_embedding_model()`, `get_embedding_dimension()`

**Orchestrator call:** `ingest.py` → `embed_chunks(chunks, show_progress=True)` converts each `ChunkRecord` into a dict and adds a `content_vector` field.

**Model loading** — `get_embedding_model()`:

1. Reads `LOCAL_EMBED_MODEL` from config (default: `"sentence-transformers/all-MiniLM-L6-v2"`).
2. Auto-detects GPU: `device = "cuda" if torch.cuda.is_available() else "cpu"`.
3. First tries `SentenceTransformer(model_name, device=device, local_files_only=True)` to load from cache.
4. If the local cache miss occurs, falls back to downloading the model (~90 MB).
5. Stores the model in a module singleton `_model` so it is loaded exactly once per process.
6. Logs a warning if the model's embedding dimension does not match `config.embedding_dimensions` (384).

**Embedding generation** — `embed_texts(texts, batch_size=32, show_progress)`:

1. Calls `model.encode(texts, batch_size=32, show_progress_bar=show_progress, convert_to_numpy=True)`.
2. Sentence-transformers internally tokenizes each text, runs it through the transformer layers, and applies mean pooling over the token embeddings to produce a single 384-dimensional vector per text.
3. Returns a list of Python float lists: `List[List[float]]`.

**The model — `all-MiniLM-L6-v2`:**

| Property | Value |
|---|---|
| Architecture | 6-layer MiniLM (distilled from a larger model) |
| Output dimensions | 384 |
| Max input tokens | 512 tokens (~1500 chars of English text) |
| Similarity metric | Cosine (matched by the HNSW config in the index) |
| Size on disk | ~90 MB |
| Inference device | CPU or CUDA GPU (auto-detected) |
| Training data | 1B+ sentence pairs |
| Normalization | Embeddings are L2-normalized by default |

**Why local instead of a cloud embedding API?**

- Zero cost: no per-token charges.
- Zero latency from network round trips.
- No rate limits.
- No API key needed.
- Fully deterministic: the same text always produces the same vector.
- Offline capable: works without internet after first model download.

**Chunk conversion** — `embed_chunks()`:

Takes a list of `ChunkRecord` dataclass instances, extracts the `content` field from each, generates all embeddings in one batched call, then returns a list of plain dicts with all chunk fields plus `content_vector`.

---

### Step 6: Upload to Azure AI Search

**File:** `search_client.py`
**Functions:** `upload_documents()`, `get_search_client()`

**Orchestrator call:** `ingest.py` → `prepare_documents_for_upload(chunks_with_embeddings, timestamp)` → `upload_documents(upload_docs, batch_size=500)`.

**Document preparation** — `prepare_documents_for_upload()`:

Takes the embedding-enriched chunk dicts and produces Azure AI Search documents with these exact fields:

```python
{
    "id":             chunk["id"],               # e.g. "procedures_pdf__chunk_0003"
    "content":        chunk["content"],           # chunk text
    "content_vector": chunk["content_vector"],    # 384 floats
    "source_file":    chunk["source_file"],       # "procedures.pdf"
    "blob_path":      chunk["blob_path"],         # "raw-docs/procedures.pdf"
    "chunk_id":       chunk["chunk_id"],           # 3
    "page_info":      chunk.get("page_info", ""), # "p5-p6"
    "created_at":     timestamp.isoformat(),      # UTC ISO 8601
}
```

**Batch upload** — `upload_documents(documents, batch_size=500)`:

1. Splits the document list into batches of 500 (default).
2. For each batch, calls `search_client.upload_documents(documents=batch)`.
3. The Azure SDK returns a per-document result. Successful and failed counts are tracked.
4. A 403 error triggers a message about the missing `Search Index Data Contributor` role.
5. Returns a summary dict: `{"total": N, "succeeded": N, "failed": N, "errors": [...]}`.

**Why 500 per batch?** Azure AI Search has a per-request payload limit. 500 documents with 384-dim vectors fits well within the 16 MB request limit.

---

### Ingestion Summary

After all steps complete, `run_ingestion()` prints a summary and returns:

```python
{
    "status": "completed",
    "duration_seconds": 42.3,
    "blobs_downloaded": 12,
    "documents_extracted": 12,
    "chunks_created": 247,
    "documents_uploaded": 247,
    "upload_failures": 0,
    "index_name": "rag-index",
    "timestamp": "2026-03-10T14:22:00+00:00",
    "total_documents_in_index": 247
}
```

---

## Retrieval Pipeline — Deep Dive

**Entry point:** `retriever.py` → `retrieve(query_text, top_k, use_hybrid, filter_expr, verbose)`

### Query Transforms (optional, `ENABLE_QUERY_TRANSFORMS=true`)

**File:** `query_transform/router.py` → `transform_query()`
**File:** `query_transform/heuristics.py` → `rewrite()`, `expand()`, `decompose()`, `step_back()`

1. `router.py` classifies the query as simple, complex, or ambiguous based on word count, presence of conjunctions, and question words.
2. Based on classification, it selects which heuristic transforms to apply.
3. **rewrite**: Normalizes phrasing (e.g., removes filler words).
4. **expand**: Adds synonym and acronym variants using a built-in dictionary (e.g., "PPE" → "personal protective equipment").
5. **decompose**: Splits conjunction queries ("X and Y") into sub-queries.
6. **step_back**: Generates a broader query for context.
7. All transforms are **non-LLM** — pure string manipulation and dictionary lookups. No API calls.

### Search

For each query variant (original + transforms), a vector or hybrid search is executed against Azure AI Search.

- **Vector search:** `embedder.embed_text(query)` → 384-dim vector → `search_client.vector_search()`.
- **Hybrid search:** Same vector + passes the query text for BM25 keyword matching.

### Fusion

**File:** `query_transform/fusion.py` → `reciprocal_rank_fusion()`

When multiple query variants produce separate result sets, they are merged using **Reciprocal Rank Fusion (RRF)** with `k=60`:

```
RRF_score(doc) = Σ  1 / (k + rank_in_list_i)
```

This gives each document a fused score based on its rank across all variant result sets. Documents appearing in multiple lists get boosted.

### Post-retrieval Processing

1. **Deduplication** — `post_retrieval/dedupe.py`: Two passes. First removes exact key duplicates. Then removes chunks where word-set overlap ≥ 80%.
2. **Diversity** — `post_retrieval/diversity.py`: Caps results per `source_file` and applies round-robin selection across sources.
3. **Reranking** (optional, `ENABLE_RERANKER=true`) — `post_retrieval/reranker.py`: Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` to re-score the top results with a more accurate (but slower) model.

### Output

**File:** `output_formatter.py` → `format_grouped_output()`

- Groups results by source file.
- Generates citation strings like `[procedures.pdf p5 chunk3]`.
- Extracts top findings algorithmically (headings, keywords by frequency — no LLM).

---

## RAG + LLM Mode — Deep Dive

**Entry point:** `answer_generator.py` → `generate_answer(query, results)`

Requires `ENABLE_LLM=true` in `.env`. Activated by `--mode rag` or `/mode` toggle in interactive mode.

### Context Assembly

**File:** `context_assembler.py` → `assemble_context(results)`

Converts retrieved chunks into labeled context blocks:

```
[Doc: procedures.pdf, Pages: p5-p6, Chunk: 3]
<chunk text here>
```

Respects `MAX_CONTEXT_CHARS` (default 12,000) to avoid exceeding the model's context window.

### Prompt Construction

**File:** `prompt_builder.py` → `build_rag_prompt(query, results)`

Returns a messages list:

```python
[
    {"role": "system", "content": "<system instructions with grounding rules>"},
    {"role": "user",   "content": "Context:\n---\n<assembled context>\n---\n\nQuestion: <query>"}
]
```

The system prompt instructs the model to only answer from provided context, cite sources, and say "I don't know" if the context is insufficient.

### LLM Inference

**File:** `llm/client.py` → `chat_completion(messages, max_tokens=1500)`

1. Splits the system message into the `instructions` parameter.
2. Converts user/assistant messages into the `input` parameter.
3. Calls `client.responses.create(model=model, input=input_messages, instructions=instructions, max_output_tokens=max_tokens)`.
4. Walks the response output tree looking for `type="message"` → `content[].type="output_text"` → `.text`.
5. Falls back to `response.output_text` if the tree walk fails.

**Why Responses API, not chat.completions?** In this Foundry environment, `chat.completions.create()` returns `400: "API operation not supported for token authentication"`. The Responses API is the only working inference path.

### Post-Generation Verification

**File:** `verify.py` → `verify_answer(answer_text, citations, context_text)`

1. **Citation coverage:** Checks that every sentence in the answer is backed by at least one citation.
2. **Grounding check:** For each answer sentence, measures word overlap with the source context. Flags sentences with very low overlap as potentially ungrounded.
3. Returns `{"verdict": "pass"|"warning", "ungrounded_claims": [...], "uncovered_segments": [...]}`.

No second LLM call is made. Verification is purely algorithmic (word-set overlap).

### Citation Parsing

**File:** `prompt_builder.py` → `parse_citations_from_answer(raw_response)`

Uses regex to extract citation markers like `[procedures.pdf p5 chunk3]` from the raw LLM response text.

---

## Web UI

**Backend:** `web/app.py` — FastAPI server on `http://127.0.0.1:8000`

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the HTML application shell |
| `/api/health` | GET | Returns `{"status": "ok"}` |
| `/api/config` | GET | Returns non-sensitive runtime settings |
| `/api/query` | POST | Runs the retrieval or RAG pipeline, returns JSON |

**Frontend:** `web/templates/index.html` + `web/static/styles.css` + `web/static/app.js`

Features:
- Mode toggle (retrieval-only / RAG)
- Hybrid / vector-only search selector
- Top-K and OData filter inputs
- Preset query chips for quick testing
- Answer panel with verification status badge
- Citation pills
- Result cards with source, page info, chunk ID, and score
- Responsive layout (works on desktop and mobile)
- Keyboard shortcut: `Ctrl+Enter` to run query

The query pipeline is lazy-loaded — the retrieval stack (including sentence-transformers) only imports when you submit your first query, not at server startup.

---

## Evaluation

**Files:** `eval/golden_questions.json`, `eval/run_eval.py`

8 golden questions with expected keyword sets, tested against the oil & gas safety document corpus.

| Metric | Result |
|---|---|
| **Overall keyword recall** | 94% |
| **Perfect recall (all keywords found)** | 6 out of 8 questions |
| **Search mode** | Vector (default) |
| **LLM used during evaluation** | None (retrieval-only) |

```powershell
py eval/run_eval.py                    # Retrieval-only evaluation
py eval/run_eval.py --mode rag         # RAG evaluation
py eval/run_eval.py --verbose --hybrid # Verbose with hybrid search
```

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
| `AZURE_TENANT_ID` | No | — | Entra tenant ID (optional, see auth section) |
| `AZURE_USE_EXPLICIT_TENANT` | No | — | Set to `true` to force the tenant ID |
| `LOCAL_EMBED_MODEL` | No | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model (HuggingFace name) |
| `CHUNK_SIZE_CHARS` | No | `1500` | Chunk size in characters |
| `CHUNK_OVERLAP_CHARS` | No | `200` | Overlap between chunks in characters |
| `TOP_K` | No | `5` | Default search results count |
| `DOWNLOAD_DIR` | No | `./data/downloads` | Local directory for blob downloads |

### Feature Flags

| Variable | Default | Description |
|---|---|---|
| `ENABLE_QUERY_TRANSFORMS` | `false` | Multi-query expansion + RRF fusion |
| `ENABLE_RERANKER` | `false` | Cross-encoder reranking (local model) |
| `ENABLE_LLM` | `false` | GPT-5.3 answer generation via Foundry |
| `LLM_MODEL_DEPLOYMENT_NAME` | `gpt-5.3-chat` | Model deployment name in Foundry |
| `MAX_CONTEXT_CHARS` | `12000` | Max context window for LLM prompts |
| `NUM_QUERY_EXPANSIONS` | `3` | Number of query expansion variants |
| `NUM_SUB_QUERIES` | `3` | Max sub-queries for decomposition |
| `TOP_K_RETRIEVE` | `30` | Search depth before post-processing |
| `TOP_K_FINAL` | `5` | Final results after post-processing |

### Config loading — `.env` override behavior

`config.py` → `load_config()` calls `load_dotenv(env_path, override=True)`. The `override=True` is critical: it ensures that the `.env` file values always win over any pre-existing system environment variables. Without this, a stale system-level `ENABLE_LLM=false` would silently override `ENABLE_LLM=true` in your `.env`.

---

## Key Design Decisions

1. **Local embeddings over cloud APIs** — `all-MiniLM-L6-v2` runs locally so ingestion does not depend on external embedding services. This reduces latency, avoids rate limits, and keeps embedding costs at zero.

2. **Responses API over chat.completions** — In this Foundry environment, the standard `chat.completions.create()` endpoint returns `400: API operation not supported for token authentication`. The Responses API (`responses.create()`) is the working path and is used throughout.

3. **Non-LLM query transforms** — Synonym expansion, step-back queries, and conjunction decomposition are all heuristic-based. This keeps retrieval improvements independent of LLM availability and avoids latency overhead.

4. **Post-generation verification** — After GPT generates an answer, `verify.py` checks citation coverage and grounding (word overlap between claims and source context). This catches hallucinated citations and ungrounded claims without a second LLM call.

5. **Feature flags over hardcoded behavior** — Query transforms, reranking, and LLM answers are all opt-in via environment variables. The pipeline works as a pure retrieval system with everything disabled.

6. **Smart break chunking** — Instead of splitting text at arbitrary character positions, the chunker searches for the nearest paragraph break, sentence ending, or clause boundary within ±100 characters of the target split point. This ensures chunks contain complete thoughts.

7. **Deterministic chunk IDs** — Chunk IDs are generated from the source filename and chunk number (`procedures_pdf__chunk_0003`), not random UUIDs. This means re-ingesting the same document produces the same IDs, enabling idempotent updates.

8. **Singleton pattern everywhere** — The credential, config, embedding model, OpenAI client, and search clients are all module-level singletons. This avoids re-creating expensive objects on every call, which matters especially for the embedding model (~90 MB in memory).

9. **Lazy imports for web startup** — The web app (`web/app.py`) does not import the retrieval stack at module level. The `from ..query import run_query` import happens inside the `/api/query` handler. This means the server starts in < 1 second, and the heavy model loading only happens on the first query.

10. **dotenv override=True** — A debugging lesson: `load_dotenv()` by default does NOT overwrite existing environment variables. If a system-level env var existed, the `.env` file would be silently ignored. We use `override=True` to prevent this.

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

### Config
- **Feature flag reads as False despite .env saying True:** Check for a system-level environment variable of the same name. The fix is `load_dotenv(override=True)`, which this project already uses.

---

## License

MIT License — feel free to use and modify.
