"""
Main ingestion pipeline orchestrator.

Coordinates the full RAG ingestion workflow:
1. Download documents from Azure Blob Storage
2. Extract text from PDFs and text files
3. Chunk text with overlap
4. Generate embeddings locally
5. Upload to Azure AI Search

Usage:
    py -m src.ingest [--force-reindex] [--prefix <blob_prefix>]
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from .config import get_config
from .blob_reader import download_all_blobs, BlobInfo
from .pdf_text import extract_text, ExtractedText
from .chunker import chunk_text, ChunkRecord
from .embedder import embed_chunks, get_embedding_dimension
from .schema import ensure_index_exists, validate_index_schema, index_exists
from .search_client import upload_documents, get_document_count

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_documents(blob_infos: List[BlobInfo]) -> List[tuple]:
    """
    Extract text from downloaded documents.
    
    Args:
        blob_infos: List of downloaded blob information
        
    Returns:
        List of (ExtractedText, blob_path) tuples
    """
    documents = []
    
    logger.info(f"Extracting text from {len(blob_infos)} documents...")
    
    for blob_info in tqdm(blob_infos, desc="Extracting"):
        try:
            extracted = extract_text(blob_info.local_path)
            
            if extracted.text.strip():
                documents.append((extracted, blob_info.blob_path))
            else:
                logger.warning(f"No text extracted from: {blob_info.name}")
                
        except Exception as e:
            logger.error(f"Failed to extract text from {blob_info.name}: {e}")
            continue
    
    logger.info(f"Successfully extracted text from {len(documents)} documents")
    return documents


def chunk_documents(documents: List[tuple]) -> List[ChunkRecord]:
    """
    Chunk all extracted documents.
    
    Args:
        documents: List of (ExtractedText, blob_path) tuples
        
    Returns:
        List of ChunkRecord objects
    """
    all_chunks = []
    
    logger.info(f"Chunking {len(documents)} documents...")
    
    for extracted, blob_path in tqdm(documents, desc="Chunking"):
        chunks = chunk_text(extracted=extracted, blob_path=blob_path)
        all_chunks.extend(chunks)
    
    logger.info(f"Created {len(all_chunks)} chunks total")
    return all_chunks


def prepare_documents_for_upload(
    chunks_with_embeddings: List[dict],
    timestamp: Optional[datetime] = None
) -> List[dict]:
    """
    Prepare chunk documents for upload to Azure AI Search.
    
    Ensures all required fields are present and adds timestamp.
    
    Args:
        chunks_with_embeddings: Chunks with content_vector field
        timestamp: Timestamp for created_at field (default: now)
        
    Returns:
        List of documents ready for upload
    """
    timestamp = timestamp or datetime.now(timezone.utc)
    
    documents = []
    for chunk in chunks_with_embeddings:
        doc = {
            "id": chunk["id"],
            "content": chunk["content"],
            "content_vector": chunk["content_vector"],
            "source_file": chunk["source_file"],
            "blob_path": chunk["blob_path"],
            "chunk_id": chunk["chunk_id"],
            "page_info": chunk.get("page_info", ""),
            "created_at": timestamp.isoformat(),
        }
        documents.append(doc)
    
    return documents


def run_ingestion(
    prefix: Optional[str] = None,
    force_reindex: bool = False,
    batch_size: int = 500
) -> dict:
    """
    Run the full ingestion pipeline.
    
    Args:
        prefix: Blob prefix filter (default from config)
        force_reindex: If True, recreate index from scratch
        batch_size: Upload batch size
        
    Returns:
        dict: Ingestion summary
    """
    config = get_config()
    start_time = datetime.now()
    
    logger.info("=" * 60)
    logger.info("RAG INGESTION PIPELINE")
    logger.info("=" * 60)
    
    # Step 1: Validate/create index
    logger.info("\n[1/5] Checking Azure AI Search index...")
    
    embedding_dim = get_embedding_dimension()
    logger.info(f"Embedding model dimension: {embedding_dim}")
    
    if force_reindex:
        logger.info("Force reindex requested - deleting and recreating index")
        from .schema import create_index
        create_index(embedding_dimensions=embedding_dim, force=True)
    elif not index_exists():
        logger.info("Index does not exist - creating new index")
        ensure_index_exists(embedding_dimensions=embedding_dim)
    else:
        # Validate existing index schema
        try:
            validate_index_schema(expected_dimensions=embedding_dim)
        except ValueError as e:
            logger.error(f"Index schema validation failed: {e}")
            logger.error("Use --force-reindex to recreate the index")
            return {"error": str(e)}
    
    # Step 2: Download documents from blob storage
    logger.info("\n[2/5] Downloading documents from Azure Blob Storage...")
    logger.info(f"Storage: {config.azure_storage_account_url}")
    logger.info(f"Container: {config.azure_storage_container}")
    
    blob_infos = download_all_blobs(prefix=prefix, show_progress=True)
    
    if not blob_infos:
        logger.warning("No documents found to process!")
        return {"error": "No documents found", "documents": 0}
    
    logger.info(f"Downloaded {len(blob_infos)} documents")
    
    # Step 3: Extract text
    logger.info("\n[3/5] Extracting text from documents...")
    
    documents = extract_documents(blob_infos)
    
    if not documents:
        logger.error("No text could be extracted from any document!")
        return {"error": "Text extraction failed", "documents": len(blob_infos)}
    
    # Step 4: Chunk and embed
    logger.info("\n[4/5] Chunking text and generating embeddings...")
    
    chunks = chunk_documents(documents)
    
    if not chunks:
        logger.error("No chunks created!")
        return {"error": "Chunking failed", "documents": len(documents)}
    
    # Generate embeddings
    chunks_with_embeddings = embed_chunks(chunks, show_progress=True)
    
    # Prepare for upload
    timestamp = datetime.now(timezone.utc)
    upload_docs = prepare_documents_for_upload(chunks_with_embeddings, timestamp)
    
    # Step 5: Upload to Azure AI Search
    logger.info("\n[5/5] Uploading to Azure AI Search...")
    logger.info(f"Index: {config.azure_search_index}")
    logger.info(f"Documents to upload: {len(upload_docs)}")
    
    upload_result = upload_documents(upload_docs, batch_size=batch_size)
    
    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    summary = {
        "status": "completed",
        "duration_seconds": round(duration, 2),
        "blobs_downloaded": len(blob_infos),
        "documents_extracted": len(documents),
        "chunks_created": len(chunks),
        "documents_uploaded": upload_result["succeeded"],
        "upload_failures": upload_result["failed"],
        "index_name": config.azure_search_index,
        "timestamp": timestamp.isoformat()
    }
    
    logger.info("\n" + "=" * 60)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Duration: {duration:.1f} seconds")
    logger.info(f"Documents processed: {len(documents)}")
    logger.info(f"Chunks created: {len(chunks)}")
    logger.info(f"Successfully uploaded: {upload_result['succeeded']}")
    
    if upload_result["failed"] > 0:
        logger.warning(f"Failed uploads: {upload_result['failed']}")
    
    # Get final document count
    try:
        total_docs = get_document_count()
        logger.info(f"Total documents in index: {total_docs}")
        summary["total_documents_in_index"] = total_docs
    except Exception as e:
        logger.warning(f"Could not get document count: {e}")
    
    return summary


def main():
    """Main entry point for the ingestion script."""
    parser = argparse.ArgumentParser(
        description="RAG Ingestion Pipeline - Ingest documents into Azure AI Search"
    )
    
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Blob prefix filter (default: from .env or no filter)"
    )
    
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Delete and recreate the index (WARNING: deletes all existing data)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Upload batch size (default: 500)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        result = run_ingestion(
            prefix=args.prefix,
            force_reindex=args.force_reindex,
            batch_size=args.batch_size
        )
        
        if "error" in result:
            logger.error(f"Ingestion failed: {result['error']}")
            sys.exit(1)
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.info("\nIngestion cancelled by user")
        sys.exit(130)
        
    except Exception as e:
        logger.exception(f"Ingestion failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
