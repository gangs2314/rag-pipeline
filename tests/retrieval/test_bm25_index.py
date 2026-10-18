"""Tests for BM25 index."""

from rag_pipeline.retrieval.bm25_index import BM25Index


def test_bm25_build_and_search():
    """Test BM25 index build and search."""
    index = BM25Index()

    chunks = [
        {
            "content": "Machine learning is a field of AI",
            "metadata": {"chunk_id": "c1"},
        },
        {
            "content": "Deep learning uses neural networks",
            "metadata": {"chunk_id": "c2"},
        },
        {
            "content": "Natural language processing with transformers",
            "metadata": {"chunk_id": "c3"},
        },
    ]

    index.build(chunks)

    results = index.search("machine learning", top_k=2)

    assert len(results) == 2
    assert results[0]["metadata"]["chunk_id"] == "c1"  # Most relevant
    assert "bm25_score" in results[0]
    assert results[0]["bm25_score"] > 0


def test_bm25_empty_index():
    """Test BM25 on empty index."""
    index = BM25Index()
    results = index.search("query", top_k=5)
    assert results == []


def test_bm25_top_k_limit():
    """Test BM25 respects top_k."""
    index = BM25Index()

    chunks = [
        {"content": f"Content {i} with learning", "metadata": {"chunk_id": f"c{i}"}}
        for i in range(10)
    ]

    index.build(chunks)
    results = index.search("learning", top_k=3)

    assert len(results) == 3


def test_bm25_keyword_matching():
    """Test BM25 prioritizes keyword matches."""
    index = BM25Index()

    chunks = [
        {"content": "Python is a programming language", "metadata": {"chunk_id": "c1"}},
        {"content": "Python and Django for web development", "metadata": {"chunk_id": "c2"}},
        {"content": "Java programming with Spring framework", "metadata": {"chunk_id": "c3"}},
    ]

    index.build(chunks)
    results = index.search("Python Django", top_k=3)

    # c2 should rank highest (has both keywords)
    assert results[0]["metadata"]["chunk_id"] == "c2"


def test_bm25_case_insensitive():
    """Test BM25 is case insensitive."""
    index = BM25Index()

    chunks = [
        {"content": "Machine Learning Algorithms", "metadata": {"chunk_id": "c1"}},
        {"content": "Deep Learning Models", "metadata": {"chunk_id": "c2"}},
    ]

    index.build(chunks)

    results_lower = index.search("machine", top_k=2)
    results_upper = index.search("MACHINE", top_k=2)

    assert len(results_lower) == 2
    assert len(results_upper) == 2
    assert results_lower[0]["metadata"]["chunk_id"] == results_upper[0]["metadata"]["chunk_id"]


def test_bm25_chunk_ids():
    """Test BM25 correctly handles chunk IDs."""
    index = BM25Index()

    chunks = [
        {"content": "Content A", "metadata": {"chunk_id": "custom_id_1"}},
        {"content": "Content B", "metadata": {"chunk_id": "custom_id_2"}},
    ]

    index.build(chunks)
    results = index.search("content", top_k=2)

    assert len(results) == 2
    chunk_ids = [r["metadata"]["chunk_id"] for r in results]
    assert "custom_id_1" in chunk_ids
    assert "custom_id_2" in chunk_ids


def test_bm25_no_matches():
    """Test BM25 with no keyword matches."""
    index = BM25Index()

    chunks = [
        {"content": "Python programming", "metadata": {"chunk_id": "c1"}},
        {"content": "Java development", "metadata": {"chunk_id": "c2"}},
    ]

    index.build(chunks)
    results = index.search("Ruby", top_k=5)

    # Should still return results (BM25 assigns low scores to non-matches)
    assert len(results) <= 2


def test_bm25_score_values():
    """Test BM25 scores are reasonable."""
    index = BM25Index()

    chunks = [
        {"content": "machine learning algorithms", "metadata": {"chunk_id": "c1"}},
        {"content": "deep neural networks", "metadata": {"chunk_id": "c2"}},
    ]

    index.build(chunks)
    results = index.search("machine learning", top_k=2)

    # c1 should have highest score (contains both keywords)
    assert results[0]["metadata"]["chunk_id"] == "c1"
    assert results[0]["bm25_score"] >= results[1]["bm25_score"]
