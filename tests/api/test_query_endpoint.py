"""Tests for /query endpoint with reranking."""

from fastapi.testclient import TestClient

from rag_pipeline.api import app


def test_query_accepts_rerank_param():
    """Test /query accepts rerank parameter."""
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"query": "machine learning", "top_k": 5, "rerank": False},
    )

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert data["query"] == "machine learning"


def test_query_rerank_true():
    """Test /query with rerank=True."""
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"query": "machine learning", "top_k": 5, "rerank": True},
    )

    assert response.status_code == 200
    data = response.json()
    assert "results" in data


def test_query_search_modes():
    """Test that search modes still work."""
    client = TestClient(app)

    for mode in ["semantic", "keyword", "hybrid"]:
        response = client.post(
            "/query",
            json={"query": "test", "top_k": 5, "mode": mode},
        )

        assert response.status_code == 200


def test_query_backwards_compatible():
    """Test that /query works without new parameters."""
    client = TestClient(app)

    response = client.post(
        "/query",
        json={"query": "test"},
    )

    assert response.status_code == 200


