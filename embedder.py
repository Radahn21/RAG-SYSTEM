"""
Local embedding module.

Provides functionality to generate embeddings locally using
sentence-transformers models. No cloud embedding calls required.
"""

import logging
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import get_config

logger = logging.getLogger(__name__)

# Global model instance (singleton)
_model: Optional[SentenceTransformer] = None
_model_name: Optional[str] = None


def get_embedding_model() -> SentenceTransformer:
    """
    Get or create the singleton embedding model.
    
    Loads the model specified in LOCAL_EMBED_MODEL environment variable.
    Default is 'sentence-transformers/all-MiniLM-L6-v2' which produces
    384-dimensional vectors.
    
    Returns:
        SentenceTransformer: The loaded embedding model
    """
    global _model, _model_name
    
    config = get_config()
    
    # Check if we need to load a new model
    if _model is None or _model_name != config.local_embed_model:
        logger.info(f"Loading embedding model: {config.local_embed_model}")
        
        # Detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        try:
            _model = SentenceTransformer(
                config.local_embed_model, 
                device=device,
                local_files_only=True
            )
        except Exception as e:
            logger.warning(f"Could not load model from cache: {e}")
            logger.info("Attempting to download model (may fail behind proxy)...")
            _model = SentenceTransformer(config.local_embed_model, device=device)
        
        _model_name = config.local_embed_model
        
        # Log model info
        embedding_dim = _model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {embedding_dim}")
        
        # Verify dimension matches config
        if embedding_dim != config.embedding_dimensions:
            logger.warning(
                f"Model produces {embedding_dim}-dim vectors, but config expects "
                f"{config.embedding_dimensions}. Update config.embedding_dimensions "
                f"or ensure your Azure AI Search index uses {embedding_dim} dimensions."
            )
    
    return _model


def get_embedding_dimension() -> int:
    """
    Get the embedding dimension of the loaded model.
    
    Returns:
        int: Embedding vector dimension
    """
    model = get_embedding_model()
    return model.get_sentence_embedding_dimension()


def embed_text(text: str) -> List[float]:
    """
    Generate embedding for a single text string.
    
    Args:
        text: Text to embed
        
    Returns:
        List[float]: Embedding vector
    """
    model = get_embedding_model()
    
    # Encode and convert to list
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_texts(
    texts: List[str],
    batch_size: int = 32,
    show_progress: bool = False
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in batches.
    
    Args:
        texts: List of texts to embed
        batch_size: Number of texts to process at once
        show_progress: Show progress bar
        
    Returns:
        List[List[float]]: List of embedding vectors
    """
    if not texts:
        return []
    
    model = get_embedding_model()
    
    logger.info(f"Embedding {len(texts)} texts (batch_size={batch_size})")
    
    # Encode all texts
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True
    )
    
    # Convert to list of lists
    return [emb.tolist() for emb in embeddings]


def embed_chunks(
    chunks: List[dict],
    content_field: str = "content",
    batch_size: int = 32,
    show_progress: bool = True
) -> List[dict]:
    """
    Add embeddings to chunk dictionaries.
    
    This function takes a list of chunk dictionaries (or ChunkRecord objects)
    and adds a 'content_vector' field with the embedding.
    
    Args:
        chunks: List of chunk dictionaries with 'content' field
        content_field: Name of the field containing text to embed
        batch_size: Batch size for embedding
        show_progress: Show progress bar
        
    Returns:
        List[dict]: Chunks with 'content_vector' field added
    """
    if not chunks:
        return []
    
    # Extract texts
    texts = []
    for chunk in chunks:
        if hasattr(chunk, content_field):
            texts.append(getattr(chunk, content_field))
        elif isinstance(chunk, dict) and content_field in chunk:
            texts.append(chunk[content_field])
        else:
            raise ValueError(f"Chunk missing '{content_field}' field: {chunk}")
    
    # Generate embeddings
    logger.info(f"Generating embeddings for {len(texts)} chunks...")
    
    embeddings = embed_texts(texts, batch_size=batch_size, show_progress=show_progress)
    
    # Add embeddings to chunks
    result = []
    for chunk, embedding in zip(chunks, embeddings):
        if hasattr(chunk, "__dict__"):
            # Convert dataclass to dict
            chunk_dict = {
                "id": chunk.id,
                "content": chunk.content,
                "source_file": chunk.source_file,
                "blob_path": chunk.blob_path,
                "chunk_id": chunk.chunk_id,
                "page_info": chunk.page_info,
                "content_vector": embedding
            }
        else:
            chunk_dict = dict(chunk)
            chunk_dict["content_vector"] = embedding
        
        result.append(chunk_dict)
    
    logger.info(f"Added embeddings to {len(result)} chunks")
    return result


if __name__ == "__main__":
    # Test embedding
    logging.basicConfig(level=logging.INFO)
    
    print("Testing local embedding model...")
    print("=" * 50)
    
    # Load model
    model = get_embedding_model()
    dim = get_embedding_dimension()
    print(f"Model loaded. Embedding dimension: {dim}")
    
    # Test single embedding
    test_text = "This is a test sentence for embedding."
    embedding = embed_text(test_text)
    
    print(f"\nTest text: '{test_text}'")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")
    
    # Test batch embedding
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Azure AI Search provides powerful search capabilities.",
        "Python is a versatile programming language.",
        "RAG systems combine retrieval with generation."
    ]
    
    print(f"\nEmbedding {len(test_texts)} texts...")
    embeddings = embed_texts(test_texts, show_progress=True)
    
    print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
    
    # Test similarity
    import numpy as np
    
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    print("\nSimilarity matrix:")
    for i, text_i in enumerate(test_texts):
        sims = [cosine_similarity(embeddings[i], embeddings[j]) for j in range(len(test_texts))]
        print(f"  {i}: {[f'{s:.2f}' for s in sims]}")
