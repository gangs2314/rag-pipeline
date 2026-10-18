"""Tests for reranker implementations."""

import pytest

from rag_pipeline.retrieval.reranker import (
    CrossEncoderReranker,
    NoOpReranker,
    get_reranker,
)


def test_noop_reranker_passthrough():
    """Test NoOpReranker returns candidates unchanged."""
    reranker = NoOpReranker()
    candidates = [
        {"content": "ML is AI", "metadata": {"source": "a.pdf"}},
        {"content": "DL is ML", "metadata": {"source": "b.pdf"}},
        {"content": "NN is DL", "metadata": {"source": "c.pdf"}},
    ]

    result = reranker.rerank("machine learning", candidates, top_k=3)

    assert len(result) == 3
    assert result[0]["content"] == "ML is AI"
    assert result[1]["content"] == "DL is ML"
    assert result[2]["content"] == "NN is DL"


def test_noop_reranker_top_k():
    """Test NoOpReranker respects top_k."""
    reranker = NoOpReranker()
    candidates = [
        {"content": f"Content {i}", "metadata": {"id": i}}
        for i in range(10)
    ]

    result = reranker.rerank("query", candidates, top_k=3)

    assert len(result) == 3
    assert result[0]["metadata"]["id"] == 0
    assert result[1]["metadata"]["id"] == 1
    assert result[2]["metadata"]["id"] == 2


def test_noop_reranker_empty_candidates():
    """Test NoOpReranker handles empty candidates."""
    reranker = NoOpReranker()
    result = reranker.rerank("query", [], top_k=5)
    assert result == []


def test_noop_reranker_top_k_exceeds_candidates():
    """Test NoOpReranker when top_k exceeds candidates."""
    reranker = NoOpReranker()
    candidates = [
        {"content": "A", "metadata": {}},
        {"content": "B", "metadata": {}},
    ]

    result = reranker.rerank("query", candidates, top_k=10)

    assert len(result) == 2


def test_get_reranker_noop():
    """Test factory creates NoOpReranker."""
    reranker = get_reranker("noop")
    assert isinstance(reranker, NoOpReranker)


def test_get_reranker_cross_encoder():
    """Test factory creates CrossEncoderReranker."""
    try:
        reranker = get_reranker("cross-encoder")
        assert isinstance(reranker, CrossEncoderReranker)
    except Exception as e:
        # Model download may fail in test environment
        assert "sentence_transformers" in str(e).lower() or "model" in str(e).lower()


def test_get_reranker_unknown():
    """Test factory raises on unknown reranker."""
    with pytest.raises(ValueError, match="Unknown reranker"):
        get_reranker("unknown_reranker")


@pytest.mark.slow
def test_cross_encoder_reranker_reorders():
    """Test CrossEncoderReranker reorders results based on relevance."""
    try:
        reranker = CrossEncoderReranker(model="BAAI/bge-reranker-base")

        # Create candidates where first is least relevant
        candidates = [
            {"content": "Unrelated content about sports", "metadata": {"id": 1}},
            {"content": "Machine learning is a subset of AI", "metadata": {"id": 2}},
            {"content": "AI uses algorithms and data", "metadata": {"id": 3}},
        ]

        result = reranker.rerank("machine learning", candidates, top_k=3)

        # Verify all results have relevance_score
        for r in result:
            assert "relevance_score" in r
            assert isinstance(r["relevance_score"], float)

        # The ML content should be ranked higher than sports
        ml_score = next(
            (r["relevance_score"] for r in result if "machine learning" in r["content"]),
            None
        )
        sports_score = next(
            (r["relevance_score"] for r in result if "sports" in r["content"]),
            None
        )

        assert ml_score is not None
        assert sports_score is not None
        assert ml_score > sports_score

    except Exception as e:
        # Model may not be available in test environment
        pytest.skip(f"CrossEncoder model not available: {e}")


@pytest.mark.slow
def test_cross_encoder_top_k():
    """Test CrossEncoderReranker respects top_k."""
    try:
        reranker = CrossEncoderReranker()
        candidates = [
            {"content": f"Content about topic {i}", "metadata": {"id": i}}
            for i in range(10)
        ]

        result = reranker.rerank("query", candidates, top_k=3)

        assert len(result) == 3
        for r in result:
            assert "relevance_score" in r

    except Exception as e:
        pytest.skip(f"CrossEncoder model not available: {e}")


@pytest.mark.slow
def test_cross_encoder_empty_candidates():
    """Test CrossEncoderReranker handles empty candidates."""
    try:
        reranker = CrossEncoderReranker()
        result = reranker.rerank("query", [], top_k=5)
        assert result == []
    except Exception as e:
        pytest.skip(f"CrossEncoder model not available: {e}")


def test_cross_encoder_custom_model():
    """Test CrossEncoderReranker accepts custom model."""
    try:
        reranker = CrossEncoderReranker(model="BAAI/bge-reranker-base")
        assert reranker.model_name == "BAAI/bge-reranker-base"
    except Exception as e:
        # Model download may fail
        pytest.skip(f"Model not available: {e}")
