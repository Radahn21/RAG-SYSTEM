"""Quick test to verify pipeline components work."""
import sys

def log(msg):
    print(msg, flush=True)

log("=" * 50)
log("PIPELINE TEST")
log("=" * 50)

# Step 1: Config
log("\n[1] Loading config...")
from src.config import get_config
config = get_config()
log(f"    Storage: {config.azure_storage_account_url}")
log(f"    Container: {config.azure_storage_container}")
log("    OK!")

# Step 2: List blobs
log("\n[2] Listing blobs...")
from src.blob_reader import list_blobs
blobs = list(list_blobs())
log(f"    Found {len(blobs)} files")
for b in blobs[:5]:
    log(f"    - {b['name']}")
if len(blobs) > 5:
    log(f"    ... and {len(blobs) - 5} more")

if not blobs:
    log("\n    WARNING: No files found! Check container name and prefix.")
    sys.exit(1)

# Step 3: Download first blob
log("\n[3] Testing download...")
from src.blob_reader import download_blob
first_blob = blobs[0]
info = download_blob(first_blob['name'])
log(f"    Downloaded: {info.local_path}")
log(f"    Size: {info.size_bytes} bytes")

# Step 4: Extract text
log("\n[4] Extracting text...")
from src.pdf_text import extract_text
extracted = extract_text(info.local_path)
log(f"    Extracted {len(extracted.text)} characters")
log(f"    First 100 chars: {extracted.text[:100]}...")

# Step 5: Chunk
log("\n[5] Chunking...")
from src.chunker import chunk_text
chunks = chunk_text(extracted, info.blob_path)
log(f"    Created {len(chunks)} chunks")

# Step 6: Test embedding (just 1 chunk)
log("\n[6] Testing embedding...")
from src.embedder import embed_text, get_embedding_dimension
dim = get_embedding_dimension()
log(f"    Model dimension: {dim}")
emb = embed_text(chunks[0].content[:500])
log(f"    Embedding generated: {len(emb)} floats")

# Step 7: Test search client
log("\n[7] Testing search client...")
from src.search_client import get_document_count
count = get_document_count()
log(f"    Current documents in index: {count}")

log("\n" + "=" * 50)
log("ALL TESTS PASSED!")
log("=" * 50)
log("\nYou can now run: py -m src.ingest")
