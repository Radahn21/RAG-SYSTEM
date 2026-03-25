"""
Answer verification and quality gates.

Checks that the LLM answer is actually derived from the retrieved context,
not hallucinated or answering an off-topic question.
"""

import re
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

# Phrases the LLM uses when it correctly declines to answer
_DECLINE_PHRASES = [
    "don't have enough information",
    "do not have enough information",
    "cannot answer",
    "not enough information",
    "no relevant information",
    "not covered in",
    "not mentioned in",
    "no information available",
    "cannot find",
    "outside the scope",
    "not related to",
    "does not contain",
    "i'm unable to",
    "unable to answer",
    "beyond the scope",
]


def _is_decline_answer(answer: str) -> bool:
    """Check if the LLM correctly declined to answer."""
    lower = answer.lower()
    return any(phrase in lower for phrase in _DECLINE_PHRASES)


def _compute_answer_context_overlap(answer: str, context_text: str) -> float:
    """
    Compute what fraction of the answer's content words appear in the context.

    Returns a ratio between 0.0 and 1.0.
    """
    stopwords = {
        "the", "and", "for", "are", "but", "not", "you", "all", "can",
        "had", "her", "was", "one", "our", "out", "has", "have", "been",
        "this", "that", "with", "from", "they", "will", "would", "there",
        "their", "what", "about", "which", "when", "make", "like",
        "could", "into", "than", "other", "more", "some", "very", "just",
        "also", "such", "each", "these", "those", "does", "only", "any",
        "may", "shall", "should", "must", "need", "use", "used", "using",
        "based", "following", "answer", "question", "information",
        "document", "documents", "context", "provided", "above", "below",
        "include", "includes", "included", "including", "require",
        "required", "requires", "ensure", "ensures", "according",
        "provide", "provides", "states", "state", "recommend",
        "recommended", "recommends", "refer", "refers", "section",
        "key", "main", "important", "specific", "related", "relevant",
        "described", "outlined", "mentioned", "noted", "listed",
        "summary", "overview", "details", "detail", "example",
    }

    context_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", context_text.lower()))
    answer_words = set(re.findall(r"\b[a-zA-Z]{3,}\b", answer.lower())) - stopwords

    if len(answer_words) < 3:
        return 1.0  # Too short to evaluate meaningfully

    overlap = len(answer_words & context_words) / len(answer_words)
    return overlap


def verify_answer(
    answer: str,
    citations: List[str],
    context_text: str,
) -> Dict:
    """
    Verify that the answer is grounded in the retrieved context.

    Logic:
    - If the LLM declined to answer ("I don't have enough information"),
      that IS grounded — the model correctly refused.
    - If the LLM gave a substantive answer, check that at least 30% of
      its content words appear in the context. A legitimate RAG answer
      will naturally share vocabulary with its source documents.
      An off-topic answer about burgers/pizza will have near-zero overlap
      with oil & gas documents.

    Returns:
        Dict with verdict "grounded" or "not_grounded".
    """
    if not answer or not answer.strip():
        return {
            "citation_ok": True,
            "uncovered_segments": [],
            "ungrounded_claims": [],
            "verdict": "grounded",
        }

    # If the LLM correctly declined, that's a grounded response
    if _is_decline_answer(answer):
        logger.info("LLM declined to answer — marking as grounded (correct behavior)")
        return {
            "citation_ok": True,
            "uncovered_segments": [],
            "ungrounded_claims": [],
            "verdict": "grounded",
        }

    # Check overall answer-to-context word overlap
    overlap = _compute_answer_context_overlap(answer, context_text)
    logger.info(f"Answer-context word overlap: {overlap:.1%}")

    # 30% overlap is very achievable for any real RAG answer since the
    # LLM is paraphrasing from the context. Off-topic answers (burger,
    # pizza, SQL injection unrelated to docs) will score well below this.
    is_grounded = overlap >= 0.30

    if not is_grounded:
        logger.warning(
            f"Answer appears not grounded (overlap={overlap:.1%}). "
            "The answer content does not match the retrieved documents."
        )

    return {
        "citation_ok": is_grounded,
        "uncovered_segments": [],
        "ungrounded_claims": [] if is_grounded else ["Answer content does not match retrieved documents"],
        "verdict": "grounded" if is_grounded else "not_grounded",
    }
