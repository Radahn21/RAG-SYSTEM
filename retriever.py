"""
Retrieval orchestrator.

Combines query transformation, multi-query search, result fusion,
and post-retrieval processing into a single pipeline.
"""

import logging
from typing import List, Dict, Optional

from .config import get_config
from .embedder import embed_text
from .search_client import vector_search, hybrid_search

logger = logging.getLogger(__name__)


def _run_search(
    query_text: str,
    query_vector: List[float],
    top_k: int,
    use_hybrid: bool,
    filter_expr: Optional[str],
    select_fields: List[str],
) -> List[dict]:
    """Execute a single search against the index."""
    if use_hybrid:
        return hybrid_search(
            query_text=query_text,
            query_vector=query_vector,
            top_k=top_k,
            select=select_fields,
            filter_expr=filter_expr,
        )
    else:
        return vector_search(
            query_vector=query_vector,
            top_k=top_k,
            select=select_fields,
            filter_expr=filter_expr,
        )


def retrieve(
    query_text: str,
    top_k: Optional[int] = None,
    use_hybrid: bool = False,
    filter_expr: Optional[str] = None,
    enable_transforms: Optional[bool] = None,
    verbose: bool = False,
) -> Dict:
    """
    Full retrieval pipeline: transform → search → fuse → post-process.

    Args:
        query_text: User query.
        top_k: Override for top_k_retrieve (search depth).
        use_hybrid: Use hybrid search.
        filter_expr: OData filter expression.
        enable_transforms: Override ENABLE_QUERY_TRANSFORMS config.
        verbose: Print debug info.

    Returns:
        Dict with:
            results: Final list of result dicts.
            query_variants: Dict mapping variant -> raw results (for logging).
            metadata: Dict with pipeline stats.
    """
    config = get_config()
    top_k_search = top_k or config.top_k_retrieve
    top_k_final = config.top_k_final
    do_transforms = enable_transforms if enable_transforms is not None else config.enable_query_transforms

    select_fields = [
        "id", "content", "source_file", "blob_path",
        "chunk_id", "page_info", "created_at"
    ]

    query_variants: Dict[str, List[dict]] = {}

    if do_transforms:
        from .query_transform.router import transform_query
        variants = transform_query(query_text)
        if verbose:
            print(f"\n  Query variants ({len(variants)}):")
            for i, v in enumerate(variants):
                print(f"    [{i+1}] {v}")

        for variant in variants:
            variant_vector = embed_text(variant)
            results = _run_search(
                query_text=variant,
                query_vector=variant_vector,
                top_k=top_k_search,
                use_hybrid=use_hybrid,
                filter_expr=filter_expr,
                select_fields=select_fields,
            )
            query_variants[variant] = results

        # Fuse results
        from .query_transform.fusion import reciprocal_rank_fusion
        merged = reciprocal_rank_fusion(query_variants)
    else:
        # Single query path
        query_vector = embed_text(query_text)
        results = _run_search(
            query_text=query_text,
            query_vector=query_vector,
            top_k=top_k_search,
            use_hybrid=use_hybrid,
            filter_expr=filter_expr,
            select_fields=select_fields,
        )
        query_variants[query_text] = results
        # Annotate with provenance
        for r in results:
            r["retrieved_by"] = [query_text]
        merged = results

    # Post-retrieval pipeline
    from .post_retrieval.dedupe import deduplicate_chunks
    from .post_retrieval.diversity import enforce_diversity

    merged = deduplicate_chunks(merged)
    merged = enforce_diversity(merged, top_k_final=top_k_final)

    # Optional reranking
    if config.enable_reranker:
        from .post_retrieval.reranker import rerank
        merged = rerank(query_text, merged, top_n=top_k_final)
    else:
        merged = merged[:top_k_final]

    metadata = {
        "num_variants": len(query_variants),
        "total_raw_results": sum(len(v) for v in query_variants.values()),
        "final_count": len(merged),
        "transforms_enabled": do_transforms,
        "reranker_enabled": config.enable_reranker,
    }

    return {
        "results": merged,
        "query_variants": query_variants,
        "metadata": metadata,
    }
