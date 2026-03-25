"""
Configuration module for RAG Ingestion Pipeline.

Loads environment variables from .env file and provides
typed configuration values with validation and defaults.
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration settings for the RAG pipeline."""
    
    # Azure AI Search
    azure_search_endpoint: str
    azure_search_index: str
    
    # Azure Blob Storage
    azure_storage_account_url: str
    azure_storage_container: str
    azure_storage_prefix: str
    
    # Local Embedding Model
    local_embed_model: str
    
    # Chunking
    chunk_size_chars: int
    chunk_overlap_chars: int
    
    # Query
    top_k: int
    
    # Local paths
    download_dir: Path
    
    # Embedding dimensions (determined by model)
    embedding_dimensions: int = 384  # all-MiniLM-L6-v2 produces 384-dim vectors

    # --- Retrieval improvements ---
    enable_query_transforms: bool = False
    enable_reranker: bool = False
    enable_llm: bool = False
    llm_model_deployment_name: str = "gpt-5.3-chat"
    max_context_chars: int = 12000
    num_query_expansions: int = 3
    num_sub_queries: int = 3
    top_k_retrieve: int = 30
    top_k_final: int = 5


def _get_required_env(key: str) -> str:
    """Get a required environment variable or raise an error."""
    value = os.getenv(key)
    if not value:
        raise ValueError(
            f"Missing required environment variable: {key}\n"
            f"Please set it in your .env file or environment."
        )
    return value


def _get_optional_env(key: str, default: str) -> str:
    """Get an optional environment variable with a default value."""
    return os.getenv(key, default)


def load_config() -> Config:
    """
    Load configuration from environment variables.
    
    Looks for a .env file in the current directory or parent directories.
    Required variables will raise an error if not set.
    
    Returns:
        Config: Validated configuration object
        
    Raises:
        ValueError: If required environment variables are missing
    """
    # Load .env file from current directory or parent
    env_paths = [
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
        Path(__file__).parent.parent / ".env",
    ]
    
    env_loaded = False
    for env_path in env_paths:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            logger.info(f"Loaded environment from: {env_path}")
            env_loaded = True
            break
    
    if not env_loaded:
        logger.warning(
            "No .env file found. Using environment variables directly. "
            "Copy .env.example to .env and configure your settings."
        )
    
    # Load and validate configuration
    config = Config(
        # Required Azure AI Search settings
        azure_search_endpoint=_get_required_env("AZURE_SEARCH_ENDPOINT"),
        azure_search_index=_get_optional_env("AZURE_SEARCH_INDEX", "rag-index"),
        
        # Required Azure Blob Storage settings
        azure_storage_account_url=_get_required_env("AZURE_STORAGE_ACCOUNT_URL"),
        azure_storage_container=_get_required_env("AZURE_STORAGE_CONTAINER"),
        azure_storage_prefix=_get_optional_env("AZURE_STORAGE_PREFIX", ""),
        
        # Local embedding model
        local_embed_model=_get_optional_env(
            "LOCAL_EMBED_MODEL", 
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        
        # Chunking settings
        chunk_size_chars=int(_get_optional_env("CHUNK_SIZE_CHARS", "1500")),
        chunk_overlap_chars=int(_get_optional_env("CHUNK_OVERLAP_CHARS", "200")),
        
        # Query settings
        top_k=int(_get_optional_env("TOP_K", "5")),
        
        # Local paths
        download_dir=Path(_get_optional_env("DOWNLOAD_DIR", "./data/downloads")),

        # Retrieval improvements
        enable_query_transforms=_get_optional_env("ENABLE_QUERY_TRANSFORMS", "false").lower() == "true",
        enable_reranker=_get_optional_env("ENABLE_RERANKER", "false").lower() == "true",
        enable_llm=_get_optional_env("ENABLE_LLM", "false").lower() == "true",
        llm_model_deployment_name=_get_optional_env("LLM_MODEL_DEPLOYMENT_NAME", "gpt-5.3-chat"),
        max_context_chars=int(_get_optional_env("MAX_CONTEXT_CHARS", "12000")),
        num_query_expansions=int(_get_optional_env("NUM_QUERY_EXPANSIONS", "3")),
        num_sub_queries=int(_get_optional_env("NUM_SUB_QUERIES", "3")),
        top_k_retrieve=int(_get_optional_env("TOP_K_RETRIEVE", "30")),
        top_k_final=int(_get_optional_env("TOP_K_FINAL", "5")),
    )
    
    # Validate chunking settings
    if config.chunk_overlap_chars >= config.chunk_size_chars:
        raise ValueError(
            f"CHUNK_OVERLAP_CHARS ({config.chunk_overlap_chars}) must be less than "
            f"CHUNK_SIZE_CHARS ({config.chunk_size_chars})"
        )
    
    # Create download directory if it doesn't exist
    config.download_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Configuration loaded successfully")
    logger.debug(f"  Search endpoint: {config.azure_search_endpoint}")
    logger.debug(f"  Search index: {config.azure_search_index}")
    logger.debug(f"  Storage account: {config.azure_storage_account_url}")
    logger.debug(f"  Storage container: {config.azure_storage_container}")
    logger.debug(f"  Embed model: {config.local_embed_model}")
    logger.debug(f"  Chunk size: {config.chunk_size_chars} chars")
    logger.debug(f"  Chunk overlap: {config.chunk_overlap_chars} chars")
    
    return config


# Singleton pattern - load config once
_config: Optional[Config] = None


def get_config() -> Config:
    """
    Get the singleton configuration instance.
    
    Returns:
        Config: The configuration object
    """
    global _config
    if _config is None:
        _config = load_config()
    return _config


if __name__ == "__main__":
    # Test configuration loading
    config = get_config()
    print(f"Azure Search Endpoint: {config.azure_search_endpoint}")
    print(f"Azure Search Index: {config.azure_search_index}")
    print(f"Storage Account URL: {config.azure_storage_account_url}")
    print(f"Storage Container: {config.azure_storage_container}")
    print(f"Embedding Model: {config.local_embed_model}")
    print(f"Chunk Size: {config.chunk_size_chars}")
    print(f"Chunk Overlap: {config.chunk_overlap_chars}")
    print(f"Top K: {config.top_k}")
