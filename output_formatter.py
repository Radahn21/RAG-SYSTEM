"""
Output formatting for retrieval results.

Groups results by source file, deduplicates near-identical chunks,
produces clean citations, and extracts top findings algorithmically.
"""

import re
import logging
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


def group_by_source(results: List[dict]) -> Dict[str, List[dict]]:
    """Group results by source_file, preserving order of first appearance."""
    groups: Dict[str, List[dict]] = {}
    for r in results:
        key = r.get("source_file", "Unknown")
        groups.setdefault(key, []).append(r)
    return groups


def format_citation(result: dict) -> str:
    """Produce a clean citation string: [source_file pX chunkY]."""
    source = result.get("source_file", "unknown")
    page = result.get("page_info", "")
    chunk = result.get("chunk_id", "")

    parts = [source]
    if page:
        parts.append(page)
    if chunk is not None and chunk != "":
        parts.append(f"chunk{chunk}")
    return "[" + " ".join(parts) + "]"


def format_citations(results: List[dict]) -> List[str]:
    """Produce unique citation strings for all results."""
    seen = set()
    citations = []
    for r in results:
        c = format_citation(r)
        if c not in seen:
            seen.add(c)
            citations.append(c)
    return citations


def _extract_headings(text: str) -> List[str]:
    """Extract likely headings from chunk text."""
    headings = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or len(stripped) < 4:
            continue
        # Lines that are short, title-cased or all-caps are likely headings
        if len(stripped) < 80 and (stripped.istitle() or stripped.isupper()):
            headings.append(stripped)
        # Numbered headings like "1. Overview" or "3.2 Safety"
        elif re.match(r"^\d+[\.\)]\s+\S", stripped) and len(stripped) < 100:
            headings.append(stripped)
    return headings


def _extract_keywords(texts: List[str], top_n: int = 10) -> List[str]:
    """Extract top keywords by frequency (simple stopword-filtered)."""
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "and", "but", "or",
        "nor", "not", "so", "yet", "both", "either", "neither", "each",
        "every", "all", "any", "few", "more", "most", "other", "some", "such",
        "no", "only", "own", "same", "than", "too", "very", "just", "because",
        "if", "when", "where", "how", "what", "which", "who", "whom", "this",
        "that", "these", "those", "it", "its", "they", "them", "their", "we",
        "our", "you", "your", "he", "she", "his", "her", "i", "me", "my",
        "also", "about"
    }
    counter: Counter = Counter()
    for text in texts:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        for w in words:
            if w not in stopwords:
                counter[w] += 1
    return [word for word, _ in counter.most_common(top_n)]


def extract_top_findings(results: List[dict]) -> str:
    """
    Algorithmic extraction of top findings from retrieved chunks.
    Returns a short summary with key headings and keywords (no LLM).
    """
    if not results:
        return "No results to summarize."

    texts = [r.get("content", "") for r in results]
    sources = list({r.get("source_file", "Unknown") for r in results})

    # Collect headings from all chunks
    all_headings = []
    for text in texts:
        all_headings.extend(_extract_headings(text))
    unique_headings = list(dict.fromkeys(all_headings))[:8]

    # Top keywords
    keywords = _extract_keywords(texts, top_n=10)

    lines = ["--- Top Findings (algorithmic) ---"]
    lines.append(f"Sources: {len(sources)} document(s)")
    if unique_headings:
        lines.append("Key sections:")
        for h in unique_headings:
            lines.append(f"  - {h}")
    if keywords:
        lines.append(f"Top keywords: {', '.join(keywords)}")
    return "\n".join(lines)


def deduplicate_overlapping(results: List[dict], overlap_threshold: float = 0.8) -> List[dict]:
    """
    Remove near-identical overlapping chunks.
    Keeps the higher-scored result when two chunks overlap significantly.
    """
    if not results:
        return results

    def _word_set(text: str) -> set:
        return set(text.lower().split())

    kept: List[dict] = []
    for r in results:
        r_words = _word_set(r.get("content", ""))
        is_dup = False
        for k in kept:
            k_words = _word_set(k.get("content", ""))
            if not r_words or not k_words:
                continue
            overlap = len(r_words & k_words) / min(len(r_words), len(k_words))
            if overlap >= overlap_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append(r)
    return kept


def format_grouped_output(results: List[dict], show_scores: bool = True) -> str:
    """
    Format results grouped by source file with citations.
    """
    if not results:
        return "No results found."

    deduped = deduplicate_overlapping(results)
    groups = group_by_source(deduped)

    lines = []

    # Top findings summary
    lines.append(extract_top_findings(deduped))
    lines.append("")

    # Citations
    citations = format_citations(deduped)
    lines.append(f"Citations ({len(citations)}):")
    for c in citations:
        lines.append(f"  {c}")
    lines.append("")

    # Grouped results
    for source_file, chunks in groups.items():
        lines.append(f"{'=' * 60}")
        lines.append(f"Source: {source_file}  ({len(chunks)} chunk(s))")
        lines.append(f"{'=' * 60}")

        for i, chunk in enumerate(chunks, 1):
            score = chunk.get("@search.score", 0)
            page_info = chunk.get("page_info", "")
            chunk_id = chunk.get("chunk_id", "")

            meta_parts = []
            if page_info:
                meta_parts.append(f"Pages: {page_info}")
            if chunk_id is not None and chunk_id != "":
                meta_parts.append(f"Chunk: {chunk_id}")
            if show_scores:
                meta_parts.append(f"Score: {score:.4f}")

            lines.append(f"\n  [{i}] {' | '.join(meta_parts)}")
            content = chunk.get("content", "")
            max_len = 500
            if len(content) > max_len:
                content = content[:max_len] + "..."
            # Indent content
            for cline in content.split("\n"):
                lines.append(f"      {cline}")

        lines.append("")

    return "\n".join(lines)
