"""
RAG Ingestion Pipeline for Azure AI Search.

This package provides tools for:
- Reading documents from Azure Blob Storage
- Extracting text from PDFs and text files
- Chunking text with overlap
- Generating embeddings locally using sentence-transformers
- Uploading documents to Azure AI Search
- Querying the search index with vector/hybrid search
"""

__version__ = "1.0.0"
