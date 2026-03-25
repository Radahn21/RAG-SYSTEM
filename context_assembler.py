"""
Context assembler for RAG pipeline.

Assembles retrieved chunks into clean context blocks suitable for
LLM prompts, with enforced max character limit.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional

from .config import get_config

logger = logging.getLogger(__name__)


@dataclass
class ContextBlock:
    """A single context block with metadata."""
    source_file: str
    page_info: str
    chunk_id: str
    content: str
    score: float


def build_context_blocks(results: List[dict]) -> List[ContextBlock]:
    """Convert search results into ContextBlock objects."""
    blocks = []
    for r in results:
        blocks.append(ContextBlock(
            source_file=r.get("source_file", "Unknown"),
            page_info=r.get("page_info", ""),
            chunk_id=str(r.get("chunk_id", "")),
            content=r.get("content", ""),
            score=r.get("@search.score", 0.0),
        ))
    return blocks


def format_block(block: ContextBlock) -> str:
    """Format a single context block for prompt injection."""
    meta_parts = [f"Doc: {block.source_file}"]
    if block.page_info:
        meta_parts.append(f"Pages: {block.page_info}")
    if block.chunk_id:
        meta_parts.append(f"Chunk: {block.chunk_id}")
    header = "[" + ", ".join(meta_parts) + "]"
    return f"{header}\n{block.content}"


def assemble_context(
    results: List[dict],
    max_chars: Optional[int] = None,
) -> str:
    """
    Assemble retrieved chunks into a context string for LLM prompts.

    Args:
        results: Search results with content and metadata.
        max_chars: Maximum character length. Defaults to config MAX_CONTEXT_CHARS.

    Returns:
        Assembled context string with formatted blocks.
    """
    config = get_config()
    max_chars = max_chars or config.max_context_chars

    blocks = build_context_blocks(results)
    # Sort by score descending so highest-relevance blocks are included first
    blocks.sort(key=lambda b: b.score, reverse=True)

    assembled_parts: List[str] = []
    total_chars = 0

    for block in blocks:
        formatted = format_block(block)
        if total_chars + len(formatted) + 2 > max_chars:
            # Try to fit a truncated version
            remaining = max_chars - total_chars - 2
            if remaining > 200:
                truncated_content = block.content[:remaining - 100] + "..."
                truncated_block = ContextBlock(
                    source_file=block.source_file,
                    page_info=block.page_info,
                    chunk_id=block.chunk_id,
                    content=truncated_content,
                    score=block.score,
                )
                assembled_parts.append(format_block(truncated_block))
            break
        assembled_parts.append(formatted)
        total_chars += len(formatted) + 2  # +2 for separator newlines

    context = "\n\n".join(assembled_parts)
    logger.info(f"Assembled context: {len(context)} chars from {len(assembled_parts)} blocks (limit: {max_chars})")
    return context
