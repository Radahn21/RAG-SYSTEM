"""
Text chunking module.

Provides functionality to split text into overlapping chunks
for embedding and indexing.
"""

import re
import logging
import hashlib
from dataclasses import dataclass
from typing import List, Optional

from .config import get_config
from .pdf_text import ExtractedText, get_page_range_for_position

logger = logging.getLogger(__name__)


@dataclass
class ChunkRecord:
    """A single chunk of text ready for indexing."""
    id: str                  # Unique ID: "<filename>#chunk<0001>"
    content: str             # The chunk text
    source_file: str         # Original filename
    blob_path: str           # Full blob path
    chunk_id: int            # Chunk number (0-indexed)
    page_info: str           # Page range (e.g., "p3-p5" or "")
    char_start: int          # Start position in original text
    char_end: int            # End position in original text


def sanitize_id(text: str) -> str:
    """
    Sanitize a string to be used as a document ID.
    
    Azure AI Search document IDs must be URL-safe and can contain
    only letters, numbers, dashes, underscores, and equal signs.
    
    Args:
        text: Raw string to sanitize
        
    Returns:
        str: Sanitized ID string
    """
    # Replace common problematic characters
    sanitized = text.replace("/", "_").replace("\\", "_")
    sanitized = sanitized.replace(" ", "_").replace(".", "_")
    
    # Remove any remaining non-allowed characters
    sanitized = re.sub(r"[^a-zA-Z0-9_\-=]", "", sanitized)
    
    # Ensure it's not empty and not too long
    if not sanitized:
        sanitized = hashlib.md5(text.encode()).hexdigest()[:16]
    
    # Azure has a max key length of 1024, but keep it reasonable
    return sanitized[:200]


def generate_chunk_id(source_file: str, chunk_num: int) -> str:
    """
    Generate a unique, deterministic chunk ID.
    
    Args:
        source_file: Source filename  
        chunk_num: Chunk number (0-indexed)
        
    Returns:
        str: Unique chunk ID
    """
    base = sanitize_id(source_file)
    return f"{base}__chunk_{chunk_num:04d}"


def find_best_break_point(text: str, target_pos: int, search_range: int = 100) -> int:
    """
    Find the best position to break text near the target position.
    
    Prefers breaking at:
    1. Paragraph breaks (double newline)
    2. Sentence endings (. ! ?)
    3. Clause breaks (comma, semicolon, colon)
    4. Word boundaries (space)
    
    Args:
        text: The text to break
        target_pos: Target position to break near
        search_range: How far to search from target
        
    Returns:
        int: Best break position
    """
    if target_pos >= len(text):
        return len(text)
    
    # Define search boundaries
    search_start = max(0, target_pos - search_range)
    search_end = min(len(text), target_pos + search_range)
    search_text = text[search_start:search_end]
    
    # Priority ordered break patterns
    patterns = [
        r"\n\n",           # Paragraph break
        r"[.!?]\s+",       # Sentence ending
        r"[;:]\s+",        # Clause break
        r",\s+",           # Comma
        r"\s+",            # Any whitespace
    ]
    
    best_pos = target_pos
    
    for pattern in patterns:
        matches = list(re.finditer(pattern, search_text))
        if matches:
            # Find the match closest to (but not after) target_pos
            for match in reversed(matches):
                abs_pos = search_start + match.end()
                if abs_pos <= target_pos + search_range // 2:
                    return abs_pos
            
            # If no match before target, use first match after
            first_match = matches[0]
            return search_start + first_match.end()
    
    return best_pos


def chunk_text(
    extracted: ExtractedText,
    blob_path: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    smart_breaks: bool = True
) -> List[ChunkRecord]:
    """
    Split extracted text into overlapping chunks.
    
    Args:
        extracted: The ExtractedText object to chunk
        blob_path: Full blob path for metadata
        chunk_size: Chunk size in characters (default from config)
        chunk_overlap: Overlap size in characters (default from config)
        smart_breaks: Try to break at sentence/paragraph boundaries
        
    Returns:
        List[ChunkRecord]: List of chunk records ready for embedding
    """
    config = get_config()
    
    chunk_size = chunk_size or config.chunk_size_chars
    chunk_overlap = chunk_overlap or config.chunk_overlap_chars
    
    text = extracted.text.strip()
    
    if not text:
        logger.warning(f"No text to chunk for {extracted.source_file}")
        return []
    
    chunks = []
    current_pos = 0
    chunk_num = 0
    
    while current_pos < len(text):
        # Calculate end position
        end_pos = current_pos + chunk_size
        
        # Find best break point if smart_breaks is enabled
        if smart_breaks and end_pos < len(text):
            end_pos = find_best_break_point(text, end_pos)
        
        # Ensure we don't go past the end
        end_pos = min(end_pos, len(text))
        
        # Extract chunk text
        chunk_text_content = text[current_pos:end_pos].strip()
        
        if chunk_text_content:
            # Get page info for PDFs
            page_info = get_page_range_for_position(extracted, current_pos, end_pos)
            
            chunk = ChunkRecord(
                id=generate_chunk_id(extracted.source_file, chunk_num),
                content=chunk_text_content,
                source_file=extracted.source_file,
                blob_path=blob_path,
                chunk_id=chunk_num,
                page_info=page_info,
                char_start=current_pos,
                char_end=end_pos
            )
            chunks.append(chunk)
            chunk_num += 1
        
        # Move to next position with overlap
        # If we've reached the end, break
        if end_pos >= len(text):
            break
            
        # Calculate next start position (accounting for overlap)
        next_pos = end_pos - chunk_overlap
        
        # Ensure we make progress
        if next_pos <= current_pos:
            next_pos = current_pos + chunk_size // 2
        
        current_pos = next_pos
    
    logger.info(
        f"Created {len(chunks)} chunks from {extracted.source_file} "
        f"({len(text)} chars, ~{chunk_size} chars/chunk)"
    )
    
    return chunks


def chunk_documents(
    documents: List[tuple],  # List of (ExtractedText, blob_path)
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> List[ChunkRecord]:
    """
    Chunk multiple documents.
    
    Args:
        documents: List of (ExtractedText, blob_path) tuples
        chunk_size: Chunk size in characters
        chunk_overlap: Overlap size in characters
        
    Returns:
        List[ChunkRecord]: All chunks from all documents
    """
    all_chunks = []
    
    for extracted, blob_path in documents:
        chunks = chunk_text(
            extracted=extracted,
            blob_path=blob_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        all_chunks.extend(chunks)
    
    logger.info(f"Total chunks from {len(documents)} documents: {len(all_chunks)}")
    return all_chunks


if __name__ == "__main__":
    # Test chunking
    import sys
    from pathlib import Path
    from .pdf_text import extract_text
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: py -m src.chunker <file_path>")
        sys.exit(1)
    
    file_path = Path(sys.argv[1])
    
    print(f"Testing chunking on: {file_path}")
    print("=" * 50)
    
    # Extract text
    extracted = extract_text(file_path)
    print(f"Extracted {len(extracted.text)} characters")
    
    # Chunk
    chunks = chunk_text(
        extracted=extracted,
        blob_path=f"test-container/{file_path.name}"
    )
    
    print(f"\nCreated {len(chunks)} chunks:")
    print("-" * 50)
    
    for i, chunk in enumerate(chunks[:5]):  # Show first 5
        print(f"\nChunk {chunk.chunk_id}:")
        print(f"  ID: {chunk.id}")
        print(f"  Page info: {chunk.page_info or '(none)'}")
        print(f"  Length: {len(chunk.content)} chars")
        print(f"  Preview: {chunk.content[:100]}...")
    
    if len(chunks) > 5:
        print(f"\n... and {len(chunks) - 5} more chunks")
