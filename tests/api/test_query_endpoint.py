"""Tests for /query endpoint with reranking."""

import pytest
from fastapi.testclient import TestClient

from rag_pipeline import api


class FakePipeline:
    def __init__(self):
        self.last_query = None

    def query(self, query, top_k=None, mode="semantic", rerank=False, tenant_id=None):
        self.last_query = {
            "query": query,
            "top_k": top_k,
            "mode": mode,
            "rerank": rerank,
            "tenant_id": tenant_id,
        }
        return {
            "query": query,
            "result_count": 1,
            "results": [{
                "document_id": "chunk-1",
                "content": "Relevant content",
                "metadata": {"chunk_id": "chunk-1"},
                "similarity_score": 0.9,
            }],
            "status": "success",
            "mode": mode,
            "reranked": rerank,
        }


@pytest.fixture
def client(monkeypatch):
    pipeline = FakePipeline()
    monkeypatch.setattr(api, "get_pipeline", lambda: pipeline)
    monkeypatch.setattr(api, "rate_limiter", api.RateLimiter(requests_per_minute=100))
    return TestClient(api.app), pipeline


def test_query_accepts_rerank_param(client):
    """Test /query accepts rerank parameter."""
    http_client, pipeline = client

    response = http_client.post(
        "/query",
        json={"query": "machine learning", "top_k": 5, "rerank": False},
    )

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert data["query"] == "machine learning"
    assert data["reranked"] is False
    assert pipeline.last_query["rerank"] is False


def test_query_rerank_true(client):
    """Test /query with rerank=True."""
    http_client, pipeline = client

    response = http_client.post(
        "/query",
        json={"query": "machine learning", "top_k": 5, "rerank": True},
    )

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert data["reranked"] is True
    assert pipeline.last_query["rerank"] is True


def test_query_search_modes(client):
    """Test that search modes still work."""
    http_client, pipeline = client

    for mode in ["semantic", "keyword", "hybrid"]:
        response = http_client.post(
            "/query",
            json={"query": "test", "top_k": 5, "mode": mode},
        )

        assert response.status_code == 200
        assert pipeline.last_query["mode"] == mode


def test_query_backwards_compatible(client):
    """Test that /query works without new parameters."""
    http_client, _ = client

    response = http_client.post(
        "/query",
        json={"query": "test"},
    )

    assert response.status_code == 200


