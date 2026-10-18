"""Tests for fusion algorithms."""

from rag_pipeline.retrieval.fusion import rrf_fuse, weighted_fuse


def test_rrf_fuse_single_list():
    """Test RRF with single ranked list."""
    results = [
        {"content": "A", "metadata": {"chunk_id": "c1"}},
        {"content": "B", "metadata": {"chunk_id": "c2"}},
        {"content": "C", "metadata": {"chunk_id": "c3"}},
    ]

    fused = rrf_fuse([results])

    assert len(fused) == 3
    # All should have rrf_score
    for result in fused:
        assert "rrf_score" in result
        assert result["rrf_score"] > 0


def test_rrf_fuse_multiple_lists():
    """Test RRF combines multiple ranked lists."""
    list1 = [
        {"content": "ML", "metadata": {"chunk_id": "c1"}},
        {"content": "DL", "metadata": {"chunk_id": "c2"}},
        {"content": "NN", "metadata": {"chunk_id": "c3"}},
    ]
    list2 = [
        {"content": "DL", "metadata": {"chunk_id": "c2"}},
        {"content": "NN", "metadata": {"chunk_id": "c3"}},
        {"content": "ML", "metadata": {"chunk_id": "c1"}},
    ]

    fused = rrf_fuse([list1, list2], k=60)

    # Should have all unique chunks
    assert len(fused) == 3

    # c2 and c3 appear in both, so should rank higher than c1
    # First item should be c2 or c3 (appears at rank 2 and 1)
    assert fused[0]["metadata"]["chunk_id"] in ["c2", "c3"]


def test_rrf_fuse_empty_lists():
    """Test RRF handles empty input."""
    fused = rrf_fuse([])
    assert fused == []

    fused = rrf_fuse([[], []])
    assert fused == []


def test_rrf_fuse_custom_k():
    """Test RRF with custom k parameter."""
    results = [
        {"content": "A", "metadata": {"chunk_id": "c1"}},
        {"content": "B", "metadata": {"chunk_id": "c2"}},
    ]

    fused_k60 = rrf_fuse([results], k=60)
    fused_k100 = rrf_fuse([results], k=100)

    # Scores should be different with different k
    assert fused_k60[0]["rrf_score"] != fused_k100[0]["rrf_score"]


def test_weighted_fuse_dense_only():
    """Test weighted fusion with only dense results."""
    dense = [
        {"content": "A", "metadata": {"chunk_id": "c1"}, "similarity_score": 0.9},
        {"content": "B", "metadata": {"chunk_id": "c2"}, "similarity_score": 0.8},
    ]

    fused = weighted_fuse(dense, [], alpha=0.5)

    assert len(fused) == 2
    assert fused[0]["metadata"]["chunk_id"] == "c1"  # Higher score
    assert "fused_score" in fused[0]


def test_weighted_fuse_sparse_only():
    """Test weighted fusion with only sparse results."""
    sparse = [
        {"content": "A", "metadata": {"chunk_id": "c1"}, "bm25_score": 5.0},
        {"content": "B", "metadata": {"chunk_id": "c2"}, "bm25_score": 3.0},
    ]

    fused = weighted_fuse([], sparse, alpha=0.5)

    assert len(fused) == 2
    assert fused[0]["metadata"]["chunk_id"] == "c1"
    assert "fused_score" in fused[0]


def test_weighted_fuse_combined():
    """Test weighted fusion combines dense and sparse."""
    dense = [
        {"content": "ML", "metadata": {"chunk_id": "c1"}, "similarity_score": 0.9},
        {"content": "DL", "metadata": {"chunk_id": "c2"}, "similarity_score": 0.7},
    ]
    sparse = [
        {"content": "DL", "metadata": {"chunk_id": "c2"}, "bm25_score": 10.0},
        {"content": "ML", "metadata": {"chunk_id": "c1"}, "bm25_score": 5.0},
    ]

    fused = weighted_fuse(dense, sparse, alpha=0.5)

    assert len(fused) == 2
    # All have fused_score
    for result in fused:
        assert "fused_score" in result


def test_weighted_fuse_alpha_effects():
    """Test that alpha parameter affects fusion."""
    dense = [
        {"content": "A", "metadata": {"chunk_id": "c1"}, "similarity_score": 0.9},
        {"content": "B", "metadata": {"chunk_id": "c2"}, "similarity_score": 0.5},
    ]
    sparse = [
        {"content": "B", "metadata": {"chunk_id": "c2"}, "bm25_score": 10.0},
        {"content": "A", "metadata": {"chunk_id": "c1"}, "bm25_score": 1.0},
    ]

    # High alpha: favor dense
    fused_high_alpha = weighted_fuse(dense, sparse, alpha=0.9)
    # Low alpha: favor sparse
    fused_low_alpha = weighted_fuse(dense, sparse, alpha=0.1)

    # With high alpha, c1 should rank first (high dense score)
    # With low alpha, c2 should rank first (high sparse score)
    assert fused_high_alpha[0]["metadata"]["chunk_id"] == "c1"
    assert fused_low_alpha[0]["metadata"]["chunk_id"] == "c2"


def test_weighted_fuse_normalization():
    """Test weighted fusion normalizes BM25 scores."""
    dense = [
        {"content": "A", "metadata": {"chunk_id": "c1"}, "similarity_score": 0.5},
    ]
    sparse = [
        {"content": "B", "metadata": {"chunk_id": "c2"}, "bm25_score": 100.0},
        {"content": "A", "metadata": {"chunk_id": "c1"}, "bm25_score": 50.0},
    ]

    fused = weighted_fuse(dense, sparse, alpha=0.5)

    # Both should have normalized scores
    assert all("fused_score" in r for r in fused)
    assert all(0 <= r["fused_score"] <= 1 for r in fused)
