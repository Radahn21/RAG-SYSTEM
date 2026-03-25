"""
Query interface for Azure AI Search.

Supports two modes:
  retrieval-only : Search + format results (default)
  rag            : Search + GPT-5.3 answer generation (requires ENABLE_LLM=true)

Usage:
    py -m src.query "What is the document about?"
    py -m src.query --hybrid "machine learning"
    py -m src.query --top 10 "neural networks"
    py -m src.query --mode rag "What are the OSHA requirements?"
    py -m src.query --debug "safety audit"
"""

import argparse
import json
import logging
import re
import sys
from typing import List, Optional

from .config import get_config
from .schema import index_exists, validate_index_schema
from .search_client import get_document_count
from .retriever import retrieve
from .output_formatter import format_grouped_output, format_citations
from .context_assembler import assemble_context
from .retrieval_logger import log_retrieval, print_score_distribution, print_query_variant_map

# Configure logging (less verbose for query mode)
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s"
)
logger = logging.getLogger(__name__)


def _split_multi_question(query_text: str) -> List[str]:
    """
    Split a prompt that contains multiple questions into individual questions.

    Detects questions separated by '?' followed by more text, or numbered
    patterns like '1. ... 2. ...'. Returns a list of individual questions.
    If only one question is detected, returns a single-element list.
    """
    text = query_text.strip()

    # Strategy 1: Split on '?' followed by more content
    # e.g. "What is X? What is Y? What is Z?"
    parts = re.split(r'\?\s+', text)
    # Re-add the '?' to each part (except the last which may already have it)
    questions = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        if not part.endswith('?'):
            part += '?'
        # Only keep if it looks like a real question (>10 chars)
        if len(part) > 10:
            questions.append(part)

    if len(questions) > 1:
        return questions

    # Strategy 2: Split on numbered patterns "1. ... 2. ..."
    numbered = re.split(r'\d+\.\s+', text)
    questions = [q.strip().rstrip('?') + '?' for q in numbered if q.strip() and len(q.strip()) > 10]
    if len(questions) > 1:
        return questions

    # Single question
    return [text]


def run_query(
    query_text: str,
    top_k: Optional[int] = None,
    use_hybrid: bool = False,
    filter_expr: Optional[str] = None,
    verbose: bool = False,
    debug: bool = False,
    mode: str = "retrieval-only",
) -> dict:
    """
    Run a search query through the full retrieval pipeline.

    Returns:
        Dict with 'results', 'query_variants', 'metadata',
        and optionally 'answer' when mode='rag'.
    """
    config = get_config()

    if not index_exists():
        print(f"Index '{config.azure_search_index}' does not exist.")
        print("Run 'py -m src.ingest' first to create the index and ingest documents.")
        return {"results": [], "query_variants": {}, "metadata": {}}

    try:
        validate_index_schema()
    except ValueError as e:
        print(f"Index schema error: {e}")
        return {"results": [], "query_variants": {}, "metadata": {}}

    if verbose:
        print(f"Query: \"{query_text}\"")
        print(f"Mode: {mode}")
        print(f"Search type: {'Hybrid' if use_hybrid else 'Vector'}")
        if filter_expr:
            print(f"Filter: {filter_expr}")
        print()

    # Split multi-question prompts and run separate retrievals
    sub_questions = _split_multi_question(query_text)
    if len(sub_questions) > 1:
        logger.info(f"Multi-question detected: {len(sub_questions)} sub-questions")

    all_results = []
    all_variants = {}
    all_metadata = {"num_variants": 0, "total_raw_results": 0, "final_count": 0}
    seen_ids = set()

    for sq in sub_questions:
        sq_result = retrieve(
            query_text=sq,
            top_k=top_k,
            use_hybrid=use_hybrid,
            filter_expr=filter_expr,
            verbose=verbose,
        )
        # Merge results, deduplicating by chunk id
        for r in sq_result["results"]:
            rid = r.get("id", id(r))
            if rid not in seen_ids:
                seen_ids.add(rid)
                all_results.append(r)
        all_variants.update(sq_result["query_variants"])
        all_metadata["num_variants"] += sq_result["metadata"].get("num_variants", 0)
        all_metadata["total_raw_results"] += sq_result["metadata"].get("total_raw_results", 0)

    results = all_results
    query_variants = all_variants
    all_metadata["final_count"] = len(results)
    all_metadata["sub_questions"] = len(sub_questions)
    metadata = all_metadata

    pipeline_result = {
        "results": results,
        "query_variants": query_variants,
        "metadata": metadata,
    }

    # Debug output
    if debug and results:
        print_score_distribution(results)
        print_query_variant_map(query_variants)
        log_path = log_retrieval(query_text, results, query_variants)
        print(f"\n  Retrieval log saved to: {log_path}")

    # RAG + LLM mode
    if mode == "rag" and config.enable_llm:
        from .answer_generator import generate_answer
        answer_result = generate_answer(query_text, results)
        pipeline_result["answer"] = answer_result

    return pipeline_result


def interactive_mode():
    """Run in interactive query mode."""
    config = get_config()

    print("\n" + "=" * 60)
    print("RAG Query Interface - Interactive Mode")
    print("=" * 60)
    print(f"Index: {config.azure_search_index}")
    print(f"Transforms: {'ON' if config.enable_query_transforms else 'OFF'}")
    print(f"Reranker: {'ON' if config.enable_reranker else 'OFF'}")
    print(f"LLM: {'ON' if config.enable_llm else 'OFF'}")

    try:
        doc_count = get_document_count()
        print(f"Documents: {doc_count}")
    except Exception:
        print("Documents: (unable to count)")

    print("\nCommands:")
    print("  /hybrid  - Toggle hybrid search (default: vector only)")
    print("  /top N   - Set number of results (default: 5)")
    print("  /mode    - Toggle retrieval-only / rag mode")
    print("  /debug   - Toggle debug output")
    print("  /quit    - Exit")
    print("=" * 60)

    use_hybrid = False
    mode = "rag" if config.enable_llm else "retrieval-only"
    debug = False

    while True:
        try:
            query = input(f"\nQuery [{mode}]: ").strip()

            if not query:
                continue

            if query.startswith("/"):
                if query in ("/quit", "/exit"):
                    print("Goodbye!")
                    break
                elif query == "/hybrid":
                    use_hybrid = not use_hybrid
                    print(f"Hybrid search: {'ON' if use_hybrid else 'OFF'}")
                    continue
                elif query.startswith("/top "):
                    try:
                        config.top_k_final = int(query.split()[1])
                        print(f"Top K final set to: {config.top_k_final}")
                    except (IndexError, ValueError):
                        print("Usage: /top N (e.g., /top 10)")
                    continue
                elif query == "/mode":
                    mode = "rag" if mode == "retrieval-only" else "retrieval-only"
                    print(f"Mode: {mode}")
                    continue
                elif query == "/debug":
                    debug = not debug
                    print(f"Debug: {'ON' if debug else 'OFF'}")
                    continue
                else:
                    print(f"Unknown command: {query}")
                    continue

            result = run_query(
                query_text=query,
                use_hybrid=use_hybrid,
                verbose=True,
                debug=debug,
                mode=mode,
            )

            results = result.get("results", [])
            if not results:
                print("\nNo results found.")
                continue

            # Show answer if RAG mode
            answer = result.get("answer")
            if answer and answer.get("formatted"):
                print()
                print(answer["formatted"])
                print("\nSupporting Excerpts:")

            # Always show retrieval results
            print(format_grouped_output(results))

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break


def main():
    """Main entry point for the query script."""
    parser = argparse.ArgumentParser(
        description="RAG Query Interface - Search Azure AI Search index"
    )

    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        help="Search query text. If not provided, runs in interactive mode."
    )

    parser.add_argument(
        "--hybrid", "-H",
        action="store_true",
        help="Use hybrid search (vector + keyword) instead of pure vector"
    )

    parser.add_argument(
        "--top", "-k",
        type=int,
        default=None,
        help="Number of results to return (default: from config)"
    )

    parser.add_argument(
        "--filter", "-f",
        type=str,
        default=None,
        help="OData filter expression (e.g., \"source_file eq 'document.pdf'\")"
    )

    parser.add_argument(
        "--no-scores",
        action="store_true",
        help="Hide search scores in output"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: score distribution, variant map, JSON log"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["retrieval-only", "rag"],
        default="retrieval-only",
        help="Query mode: retrieval-only (default) or rag (with LLM answer)"
    )

    args = parser.parse_args()

    if args.verbose or args.debug:
        logging.getLogger().setLevel(logging.INFO)

    if not args.query:
        interactive_mode()
        return

    try:
        result = run_query(
            query_text=args.query,
            top_k=args.top,
            use_hybrid=args.hybrid,
            filter_expr=args.filter,
            verbose=args.verbose,
            debug=args.debug,
            mode=args.mode,
        )

        results = result.get("results", [])

        if not results:
            print("No results found.")
            sys.exit(0)

        # JSON output
        if args.json:
            output = {
                "query": args.query,
                "mode": args.mode,
                "metadata": result.get("metadata", {}),
                "results": [
                    {k: v for k, v in r.items() if k != "content_vector"}
                    for r in results
                ],
            }
            answer = result.get("answer")
            if answer:
                output["answer"] = answer.get("answer", "")
                output["citations"] = answer.get("citations", [])
            print(json.dumps(output, indent=2, default=str))
            sys.exit(0)

        # Show answer if RAG mode
        answer = result.get("answer")
        if answer and answer.get("formatted"):
            print()
            print(answer["formatted"])
            print("\nSupporting Excerpts:")

        # Formatted retrieval output
        print(format_grouped_output(results, show_scores=not args.no_scores))
        sys.exit(0)

    except KeyboardInterrupt:
        print("\nQuery cancelled.")
        sys.exit(130)

    except Exception as e:
        logger.exception(f"Query failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
