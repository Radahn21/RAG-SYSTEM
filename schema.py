"""
Azure AI Search index schema module.

Defines the index schema and provides functions to create or update
the index if needed.
"""

import logging
from typing import Optional

from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
    SearchableField,
    SimpleField,
)
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError

from .config import get_config
from .search_client import get_search_index_client

logger = logging.getLogger(__name__)


def get_index_schema(
    index_name: Optional[str] = None,
    embedding_dimensions: Optional[int] = None
) -> SearchIndex:
    """
    Get the search index schema definition.
    
    Schema fields:
    - id: Document key (string)
    - content: Chunk text content (searchable)
    - content_vector: Embedding vector (384 dimensions by default)
    - source_file: Original filename (filterable, facetable)
    - blob_path: Full blob path (filterable)
    - chunk_id: Chunk number (sortable)
    - page_info: Page range for PDFs (filterable)
    - created_at: Timestamp when document was indexed (filterable, sortable)
    
    Args:
        index_name: Index name (default from config)
        embedding_dimensions: Vector dimension (default from config)
        
    Returns:
        SearchIndex: Index schema definition
    """
    config = get_config()
    
    index_name = index_name or config.azure_search_index
    embedding_dimensions = embedding_dimensions or config.embedding_dimensions
    
    # Define fields
    fields = [
        # Key field
        SimpleField(
            name="id",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True
        ),
        
        # Main content field (searchable for keyword/hybrid search)
        SearchableField(
            name="content",
            type=SearchFieldDataType.String,
            searchable=True
        ),
        
        # Vector field for embeddings
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=embedding_dimensions,
            vector_search_profile_name="default-vector-profile"
        ),
        
        # Metadata fields
        SimpleField(
            name="source_file",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True
        ),
        
        SimpleField(
            name="blob_path",
            type=SearchFieldDataType.String,
            filterable=True
        ),
        
        SimpleField(
            name="chunk_id",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True
        ),
        
        SimpleField(
            name="page_info",
            type=SearchFieldDataType.String,
            filterable=True
        ),
        
        # Timestamp field (existing in your index)
        SimpleField(
            name="created_at",
            type=SearchFieldDataType.DateTimeOffset,
            filterable=True,
            sortable=True
        ),
    ]
    
    # Vector search configuration
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="default-hnsw",
                parameters={
                    "m": 4,              # Number of bi-directional links
                    "efConstruction": 400,  # Size of dynamic candidate list for construction
                    "efSearch": 500,     # Size of dynamic candidate list for search
                    "metric": "cosine"   # Distance metric
                }
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="default-vector-profile",
                algorithm_configuration_name="default-hnsw"
            )
        ]
    )
    
    # Create index definition
    index = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search
    )
    
    return index


def index_exists(index_name: Optional[str] = None) -> bool:
    """
    Check if an index exists.
    
    Args:
        index_name: Index name to check
        
    Returns:
        bool: True if index exists
    """
    config = get_config()
    index_name = index_name or config.azure_search_index
    
    client = get_search_index_client()
    
    try:
        client.get_index(index_name)
        return True
    except ResourceNotFoundError:
        return False
    except HttpResponseError as e:
        if e.status_code == 404:
            return False
        raise


def create_index(
    index_name: Optional[str] = None,
    embedding_dimensions: Optional[int] = None,
    force: bool = False
) -> SearchIndex:
    """
    Create the search index if it doesn't exist.
    
    Args:
        index_name: Index name
        embedding_dimensions: Vector dimension
        force: If True, delete and recreate existing index
        
    Returns:
        SearchIndex: The created or existing index
        
    Raises:
        ValueError: If index exists and force=False
    """
    config = get_config()
    index_name = index_name or config.azure_search_index
    
    client = get_search_index_client()
    
    # Check if index already exists
    if index_exists(index_name):
        if force:
            logger.warning(f"Deleting existing index: {index_name}")
            client.delete_index(index_name)
        else:
            logger.info(f"Index '{index_name}' already exists")
            return client.get_index(index_name)
    
    # Create new index
    logger.info(f"Creating index: {index_name}")
    
    index_schema = get_index_schema(
        index_name=index_name,
        embedding_dimensions=embedding_dimensions
    )
    
    try:
        result = client.create_index(index_schema)
        logger.info(f"Index '{index_name}' created successfully")
        return result
        
    except HttpResponseError as e:
        if e.status_code == 403:
            logger.error(
                "Access denied creating index. Ensure your account has the "
                "'Search Service Contributor' role on the search service."
            )
        raise


def ensure_index_exists(
    index_name: Optional[str] = None,
    embedding_dimensions: Optional[int] = None
) -> bool:
    """
    Ensure the index exists, creating it if necessary.
    
    This is a safe operation that will not modify an existing index.
    
    Args:
        index_name: Index name
        embedding_dimensions: Vector dimension
        
    Returns:
        bool: True if index was created, False if it already existed
    """
    config = get_config()
    index_name = index_name or config.azure_search_index
    
    if index_exists(index_name):
        logger.info(f"Index '{index_name}' exists")
        
        # Optionally validate schema compatibility
        validate_index_schema(index_name, embedding_dimensions)
        return False
    
    # Create index
    create_index(index_name, embedding_dimensions)
    return True


def validate_index_schema(
    index_name: Optional[str] = None,
    expected_dimensions: Optional[int] = None
) -> bool:
    """
    Validate that an existing index has compatible schema.
    
    Checks:
    - Required fields exist
    - Vector field has correct dimensions
    
    Args:
        index_name: Index name to validate
        expected_dimensions: Expected vector dimensions
        
    Returns:
        bool: True if schema is compatible
        
    Raises:
        ValueError: If schema is incompatible
    """
    config = get_config()
    index_name = index_name or config.azure_search_index
    expected_dimensions = expected_dimensions or config.embedding_dimensions
    
    client = get_search_index_client()
    
    try:
        index = client.get_index(index_name)
    except ResourceNotFoundError:
        raise ValueError(f"Index '{index_name}' does not exist")
    
    # Required fields
    required_fields = {"id", "content", "content_vector"}
    existing_fields = {field.name for field in index.fields}
    
    missing = required_fields - existing_fields
    if missing:
        raise ValueError(
            f"Index '{index_name}' is missing required fields: {missing}. "
            "Please recreate the index or add the missing fields."
        )
    
    # Check vector dimensions
    for field in index.fields:
        if field.name == "content_vector":
            if hasattr(field, "vector_search_dimensions"):
                actual_dims = field.vector_search_dimensions
                if actual_dims != expected_dimensions:
                    raise ValueError(
                        f"Index vector dimension mismatch. "
                        f"Index has {actual_dims} dimensions, but model produces "
                        f"{expected_dimensions} dimensions. "
                        f"Either use a different embedding model or recreate the index."
                    )
            break
    
    logger.info(f"Index '{index_name}' schema is compatible")
    return True


def get_index_info(index_name: Optional[str] = None) -> dict:
    """
    Get information about an index.
    
    Args:
        index_name: Index name
        
    Returns:
        dict: Index information
    """
    config = get_config()
    index_name = index_name or config.azure_search_index
    
    client = get_search_index_client()
    
    try:
        index = client.get_index(index_name)
        
        # Extract field info
        fields_info = []
        for field in index.fields:
            field_info = {
                "name": field.name,
                "type": str(field.type),
                "searchable": getattr(field, "searchable", False),
                "filterable": getattr(field, "filterable", False),
                "sortable": getattr(field, "sortable", False),
            }
            
            if hasattr(field, "vector_search_dimensions"):
                field_info["vector_dimensions"] = field.vector_search_dimensions
            
            fields_info.append(field_info)
        
        return {
            "name": index.name,
            "fields": fields_info,
            "vector_search": index.vector_search is not None
        }
        
    except ResourceNotFoundError:
        return {"error": f"Index '{index_name}' not found"}


if __name__ == "__main__":
    # Test schema operations
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Azure AI Search index schema...")
    print("=" * 50)
    
    config = get_config()
    
    # Check if index exists
    exists = index_exists()
    print(f"\nIndex '{config.azure_search_index}' exists: {exists}")
    
    if exists:
        # Get index info
        info = get_index_info()
        print(f"\nIndex information:")
        print(f"  Name: {info['name']}")
        print(f"  Vector search enabled: {info['vector_search']}")
        print(f"  Fields:")
        for field in info["fields"]:
            dims = field.get("vector_dimensions", "")
            dims_str = f" ({dims}d)" if dims else ""
            print(f"    - {field['name']}: {field['type']}{dims_str}")
        
        # Validate schema
        try:
            validate_index_schema()
            print("\n✓ Schema is compatible with pipeline")
        except ValueError as e:
            print(f"\n✗ Schema validation failed: {e}")
    else:
        print("\nIndex does not exist. It will be created during ingestion.")
        print("\nExpected schema:")
        schema = get_index_schema()
        for field in schema.fields:
            print(f"  - {field.name}: {field.type}")
