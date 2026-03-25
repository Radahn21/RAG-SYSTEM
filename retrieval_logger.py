"""
Retrieval debug logging and diagnostics.

Logs retrieval results to JSON, prints score distributions,
and tracks which query variant produced which chunk.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

LOGS_DIR = Path(__file__).parent.parent / "logs"


def _ensure_logs_dir() -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return LOGS_DIR


def log_retrieval(
    query: str,
    results: List[dict],
    query_variants: Optional[Dict[str, List[dict]]] = None,
    filepath: Optional[Path] = None,
) -> Path:
    """
    Save a JSON log of retrieval results.

    Args:
        query: Original user query.
        results: Final merged results.
        query_variants: Optional dict mapping variant query -> its raw results.
        filepath: Optional explicit path. Defaults to logs/retrieval_<timestamp>.json.

    Returns:
        Path to the saved log file.
    """
    _ensure_logs_dir()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if filepath is None:
        filepath = LOGS_DIR / f"retrieval_{ts}.json"

    log_entry = {
        "query": query,
        "timestamp": ts,
        "num_results": len(results),
        "results": [
            {
                "id": r.get("id", ""),
                "source_file": r.get("source_file", ""),
                "page_info": r.get("page_info", ""),
                "chunk_id": r.get("chunk_id", ""),
                "score": r.get("@search.score", 0),
                "content_preview": r.get("content", "")[:200],
                "retrieved_by": r.get("retrieved_by", []),
            }
            for r in results
        ],
    }

    if query_variants:
        log_entry["query_variants"] = {
            variant: [
                {
                    "id": r.get("id", ""),
                    "source_file": r.get("source_file", ""),
                    "score": r.get("@search.score", 0),
                }
                for r in variant_results
            ]
            for variant, variant_results in query_variants.items()
        }

    filepath.write_text(json.dumps(log_entry, indent=2, default=str), encoding="utf-8")
    logger.info(f"Retrieval log saved to: {filepath}")
    return filepath


def print_score_distribution(results: List[dict]) -> None:
    """Print a simple score distribution of top-K results."""
    if not results:
        print("  No results to show score distribution.")
        return

    scores = [r.get("@search.score", 0) for r in results]
    print(f"\n  Score Distribution ({len(scores)} results):")
    print(f"    Max:    {max(scores):.4f}")
    print(f"    Min:    {min(scores):.4f}")
    print(f"    Mean:   {sum(scores) / len(scores):.4f}")
    if len(scores) > 1:
        median = sorted(scores)[len(scores) // 2]
        print(f"    Median: {median:.4f}")

    # Simple histogram
    print("    Histogram:")
    if max(scores) > min(scores):
        num_bins = min(5, len(scores))
        bin_width = (max(scores) - min(scores)) / num_bins
        for i in range(num_bins):
            lo = min(scores) + i * bin_width
            hi = lo + bin_width
            count = sum(1 for s in scores if lo <= s < hi or (i == num_bins - 1 and s == hi))
            bar = "#" * count
            print(f"      [{lo:.3f}-{hi:.3f}] {bar} ({count})")
    else:
        print(f"      All scores equal: {scores[0]:.4f}")


def print_query_variant_map(query_variant_results: Dict[str, List[dict]]) -> None:
    """Show which query variant retrieved which chunks."""
    if not query_variant_results:
        return

    print("\n  Query Variant Provenance:")
    for variant, results in query_variant_results.items():
        chunk_ids = [
            f"{r.get('source_file', '?')}:chunk{r.get('chunk_id', '?')}"
            for r in results[:5]
        ]
        print(f"    \"{variant}\"")
        print(f"      -> {', '.join(chunk_ids)}" + (" ..." if len(results) > 5 else ""))
