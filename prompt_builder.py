"""
Prompt construction for RAG pipeline.

Builds system/user messages with context blocks and citation instructions.
"""

import logging
from typing import List, Dict, Tuple, Optional

from .context_assembler import assemble_context

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions based ONLY on the provided context. "
    "Rules:\n"
    "1. Answer ONLY from the provided context. Do not use any external knowledge.\n"
    "2. If the context does not contain the answer, say: "
    "'I don't have enough information to answer this question.'\n"
    "3. Use bullet points for multi-part answers.\n"
    "4. Include citations after each claim using the format [Doc: filename, Pages: XX, Chunk: YY].\n"
    "5. At the end, provide a 'Citations' section listing all sources referenced.\n"
    "6. Be concise and factual."
)


def build_rag_prompt(
    query: str,
    results: List[dict],
    system_instructions: Optional[str] = None,
    max_context_chars: Optional[int] = None,
) -> List[Dict[str, str]]:
    """
    Build the full message list for RAG chat completion.

    Args:
        query: User question.
        results: Retrieved search results with content and metadata.
        system_instructions: Optional override for system prompt.
        max_context_chars: Optional max chars for context assembly.

    Returns:
        List of message dicts [{"role": ..., "content": ...}].
    """
    system_msg = system_instructions or _SYSTEM_PROMPT
    context_text = assemble_context(results, max_chars=max_context_chars)

    user_content = (
        f"Context:\n"
        f"---\n"
        f"{context_text}\n"
        f"---\n\n"
        f"Question: {query}"
    )

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_content},
    ]

    logger.info(f"Built RAG prompt: system={len(system_msg)} chars, user={len(user_content)} chars")
    return messages


def parse_citations_from_answer(answer: str) -> Tuple[str, List[str]]:
    """
    Extract citations from the LLM answer.

    Looks for [Doc: ..., Pages: ..., Chunk: ...] patterns
    and the 'Citations' section at the end.

    Returns:
        Tuple of (answer_text, list_of_citation_strings).
    """
    import re

    # Find all inline citations
    citation_pattern = r"\[Doc:\s*[^\]]+\]"
    citations = re.findall(citation_pattern, answer)
    unique_citations = list(dict.fromkeys(citations))

    return answer, unique_citations


def format_answer_output(answer: str, citations: List[str]) -> str:
    """Format the final answer with citations for display."""
    lines = []
    lines.append("=" * 60)
    lines.append("ANSWER")
    lines.append("=" * 60)
    lines.append(answer.strip())

    if citations:
        lines.append("")
        lines.append("-" * 40)
        lines.append(f"Sources ({len(citations)}):")
        for c in citations:
            lines.append(f"  {c}")

    lines.append("=" * 60)
    return "\n".join(lines)
