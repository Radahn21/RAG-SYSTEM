"""
Azure AI Search client module.

Provides SearchClient and SearchIndexClient using RBAC authentication
(InteractiveBrowserCredential). No API keys required.
"""

import logging
from typing import Optional, List, Generator, Any

from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.core.exceptions import HttpResponseError

from .config import get_config
from .auth import get_credential

logger = logging.getLogger(__name__)

# Singleton clients
_search_client: Optional[SearchClient] = None
_index_client: Optional[SearchIndexClient] = None


def get_search_index_client() -> SearchIndexClient:
    """
    Get or create the SearchIndexClient for index management.
    
    Required RBAC role: Search Service Contributor (for creating/managing indexes)
    
    Returns:
        SearchIndexClient: Client for index management operations
    """
    global _index_client
    
    if _index_client is None:
        config = get_config()
        credential = get_credential()
        
        logger.info(f"Creating SearchIndexClient for: {config.azure_search_endpoint}")
        
        _index_client = SearchIndexClient(
            endpoint=config.azure_search_endpoint,
            credential=credential
        )
    
    return _index_client


def get_search_client(index_name: Optional[str] = None) -> SearchClient:
    """
    Get or create the SearchClient for document operations.
    
    Required RBAC roles:
    - Search Index Data Contributor: For uploading documents
    - Search Index Data Reader: For querying
    
    Args:
        index_name: Index name (default from config)
        
    Returns:
        SearchClient: Client for document operations
    """
    global _search_client
    
    config = get_config()
    index_name = index_name or config.azure_search_index
    
    if _search_client is None:
        credential = get_credential()
        
        logger.info(
            f"Creating SearchClient for index '{index_name}' "
            f"at {config.azure_search_endpoint}"
        )
        
        _search_client = SearchClient(
            endpoint=config.azure_search_endpoint,
            index_name=index_name,
            credential=credential
        )
    
    return _search_client


def upload_documents(
    documents: List[dict],
    batch_size: int = 500,
    index_name: Optional[str] = None
) -> dict:
    """
    Upload documents to Azure AI Search in batches.
    """
    client = get_search_client(index_name)
    
    total = len(documents)
    succeeded = 0
    failed = 0
    errors = []
    
    logger.info(f"Uploading {total} documents in batches of {batch_size}")
    
    for i in range(0, total, batch_size):
        batch = documents[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size
        
        logger.info(f"Uploading batch {batch_num}/{total_batches} ({len(batch)} documents)")
        
        try:
            result = client.upload_documents(documents=batch)
            
            for item in result:
                if item.succeeded:
                    succeeded += 1
                else:
                    failed += 1
                    errors.append({
                        "key": item.key,
                        "error": item.error_message
                    })
                    logger.warning(f"Failed to upload document {item.key}: {item.error_message}")
                    
        except HttpResponseError as e:
            failed += len(batch)
            error_msg = f"Batch {batch_num} failed: {e.message}"
            errors.append({"batch": batch_num, "error": error_msg})
            logger.error(error_msg)
            
            if e.status_code == 403:
                logger.error(
                    "Access denied. Ensure your account has the "
                    "'Search Index Data Contributor' role on the search service."
                )
                raise
    
    summary = {
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "errors": errors[:10] if len(errors) > 10 else errors
    }
    
    logger.info(f"Upload complete: {succeeded}/{total} succeeded, {failed} failed")
    
    return summary


def delete_documents(
    document_ids: List[str],
    batch_size: int = 500,
    index_name: Optional[str] = None
) -> dict:
    """
    Delete documents from Azure AI Search by ID.
    """
    client = get_search_client(index_name)
    
    total = len(document_ids)
    succeeded = 0
    failed = 0
    
    logger.info(f"Deleting {total} documents in batches of {batch_size}")
    
    for i in range(0, total, batch_size):
        batch_ids = document_ids[i:i + batch_size]
        batch_docs = [{"id": doc_id} for doc_id in batch_ids]
        
        try:
            result = client.delete_documents(documents=batch_docs)
            
            for item in result:
                if item.succeeded:
                    succeeded += 1
                else:
                    failed += 1
                    
        except HttpResponseError as e:
            failed += len(batch_ids)
            logger.error(f"Delete batch failed: {e.message}")
    
    logger.info(f"Delete complete: {succeeded}/{total} succeeded, {failed} failed")
    
    return {"total": total, "succeeded": succeeded, "failed": failed}


def vector_search(
    query_vector: List[float],
    vector_field: str = "content_vector",
    top_k: Optional[int] = None,
    select: Optional[List[str]] = None,
    filter_expr: Optional[str] = None,
    index_name: Optional[str] = None
) -> List[dict]:
    """
    Perform pure vector search.
    """
    config = get_config()
    client = get_search_client(index_name)
    
    top_k = top_k or config.top_k
    
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields=vector_field
    )
    
    results = client.search(
        search_text=None,
        vector_queries=[vector_query],
        select=select,
        filter=filter_expr,
        top=top_k
    )
    
    return [dict(result) for result in results]


def hybrid_search(
    query_text: str,
    query_vector: List[float],
    vector_field: str = "content_vector",
    top_k: Optional[int] = None,
    select: Optional[List[str]] = None,
    filter_expr: Optional[str] = None,
    semantic_config: Optional[str] = None,
    index_name: Optional[str] = None
) -> List[dict]:
    """
    Perform hybrid search (vector + keyword).
    """
    config = get_config()
    client = get_search_client(index_name)
    
    top_k = top_k or config.top_k
    
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields=vector_field
    )
    
    search_kwargs = {
        "search_text": query_text,
        "vector_queries": [vector_query],
        "select": select,
        "filter": filter_expr,
        "top": top_k
    }
    
    if semantic_config:
        search_kwargs["query_type"] = "semantic"
        search_kwargs["semantic_configuration_name"] = semantic_config
    
    results = client.search(**search_kwargs)
    
    return [dict(result) for result in results]


def get_document_count(index_name: Optional[str] = None) -> int:
    """
    Get the number of documents in the index.
    """
    client = get_search_client(index_name)
    return client.get_document_count()


if __name__ == "__main__":
    # Test search client
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Azure AI Search connection...")
    print("=" * 50)
    
    # Test index client
    index_client = get_search_index_client()
    print("\nAvailable indexes:")
    for index in index_client.list_indexes():
        print(f"  - {index.name}")
    
    # Test search client
    config = get_config()
    search_client = get_search_client()
    
    try:
        count = get_document_count()
        print(f"\nIndex '{config.azure_search_index}' has {count} documents")
    except HttpResponseError as e:
        if e.status_code == 404:
            print(f"\nIndex '{config.azure_search_index}' does not exist yet")
        else:
            print(f"\nError: {e.message}")
