"""
Answer generation orchestrator.

Combines context assembly, prompt building, and LLM calls
to produce grounded answers with citations.
"""

import logging
from typing import Dict, List, Optional

from .config import get_config
from .prompt_builder import build_rag_prompt, parse_citations_from_answer, format_answer_output
from .llm.client import chat_completion
from .context_assembler import assemble_context
from .verify import verify_answer

logger = logging.getLogger(__name__)


def generate_answer(
    query: str,
    results: List[dict],
    temperature: float = 0.2,
    max_tokens: int = 1500,
) -> Dict:
    """
    Generate a grounded answer from retrieved results using GPT.

    Args:
        query: User question.
        results: Retrieved search results.
        temperature: LLM temperature.
        max_tokens: Max response tokens.

    Returns:
        Dict with keys:
            answer: str - formatted answer text
            citations: list[str] - extracted citations
            raw_response: str - raw LLM response
            formatted: str - display-ready output
    """
    config = get_config()

    if not config.enable_llm:
        return {
            "answer": "LLM is disabled. Set ENABLE_LLM=true to enable.",
            "citations": [],
            "raw_response": "",
            "formatted": "LLM is disabled. Set ENABLE_LLM=true in .env to generate answers.",
        }

    if not results:
        return {
            "answer": "No context available to answer the question.",
            "citations": [],
            "raw_response": "",
            "formatted": "No retrieved context — cannot generate an answer.",
        }

    # Build prompt
    messages = build_rag_prompt(query, results)

    # Call LLM
    logger.info("Generating answer via LLM...")
    raw_response = chat_completion(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Parse citations
    answer_text, citations = parse_citations_from_answer(raw_response)

    # Post-generation verification
    context_text = assemble_context(results)
    verification = verify_answer(answer_text, citations, context_text)
    logger.info(f"Verification verdict: {verification['verdict']}")

    # Format for display
    formatted = format_answer_output(answer_text, citations)

    return {
        "answer": answer_text,
        "citations": citations,
        "raw_response": raw_response,
        "formatted": formatted,
        "verification": verification,
    }
