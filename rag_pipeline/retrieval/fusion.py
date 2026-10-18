"""Fusion algorithms for combining multiple ranked lists."""


def rrf_fuse(ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion combining multiple ranked lists.

    Uses RRF formula: score = sum(1 / (k + rank)) for each document.

    Args:
        ranked_lists: List of ranked result lists.
        k: Parameter for RRF (default 60).

    Returns:
        Fused ranking as list of dicts with 'rrf_score'.
    """
    if not ranked_lists or all(not lst for lst in ranked_lists):
        return []

    # Collect all unique chunks and their ranks
    chunk_scores = {}
    chunk_data = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            # Use chunk_id as key, fall back to content
            chunk_id = chunk.get("metadata", {}).get("chunk_id") or chunk.get("content", "")

            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = 0.0
                chunk_data[chunk_id] = chunk

            # Add RRF score
            chunk_scores[chunk_id] += 1.0 / (k + rank)

    # Sort by RRF score and create result list
    sorted_chunks = sorted(
        chunk_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    results = []
    for chunk_id, rrf_score in sorted_chunks:
        result = chunk_data[chunk_id].copy()
        result["rrf_score"] = rrf_score
        results.append(result)

    return results


def weighted_fuse(
    dense_results: list[dict],
    sparse_results: list[dict],
    alpha: float = 0.5
) -> list[dict]:
    """Weighted fusion of dense and sparse retrieval results.

    Combines semantic (dense) and keyword (sparse) scores:
    score = alpha * dense_score + (1 - alpha) * sparse_score

    Args:
        dense_results: Results from dense retrieval with 'similarity_score'.
        sparse_results: Results from sparse (BM25) retrieval with 'bm25_score'.
        alpha: Weight for dense results (0-1).

    Returns:
        Fused results sorted by combined score.
    """
    if not dense_results and not sparse_results:
        return []

    # Normalize scores to 0-1 range if needed
    dense_scores = {}
    sparse_scores = {}

    # Collect dense results
    for result in dense_results:
        chunk_id = result.get("metadata", {}).get("chunk_id") or result.get("content", "")
        dense_scores[chunk_id] = result.get("similarity_score", 0.0)

    # Collect sparse results and normalize BM25 scores
    for result in sparse_results:
        chunk_id = result.get("metadata", {}).get("chunk_id") or result.get("content", "")
        sparse_scores[chunk_id] = result.get("bm25_score", 0.0)

    # Normalize BM25 scores (they can be unbounded)
    max_bm25 = max(sparse_scores.values()) if sparse_scores else 1.0
    if max_bm25 > 0:
        sparse_scores = {k: v / max_bm25 for k, v in sparse_scores.items()}

    # Combine scores
    all_chunks = set(dense_scores.keys()) | set(sparse_scores.keys())
    fused_scores = {}

    for chunk_id in all_chunks:
        dense = dense_scores.get(chunk_id, 0.0)
        sparse = sparse_scores.get(chunk_id, 0.0)
        combined = alpha * dense + (1 - alpha) * sparse
        fused_scores[chunk_id] = combined

    # Get chunk data (prefer from dense results)
    chunk_data = {}
    for result in dense_results:
        chunk_id = result.get("metadata", {}).get("chunk_id") or result.get("content", "")
        chunk_data[chunk_id] = result

    for result in sparse_results:
        chunk_id = result.get("metadata", {}).get("chunk_id") or result.get("content", "")
        if chunk_id not in chunk_data:
            chunk_data[chunk_id] = result

    # Create result list sorted by fused score
    sorted_chunks = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    results = []
    for chunk_id, fused_score in sorted_chunks:
        if chunk_id in chunk_data:
            result = chunk_data[chunk_id].copy()
            result["fused_score"] = fused_score
            results.append(result)

    return results
