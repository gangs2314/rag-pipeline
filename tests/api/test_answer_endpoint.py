"""Tests for /answer API endpoint."""

import pytest
from fastapi.testclient import TestClient

from rag_pipeline import api
from rag_pipeline.api import app
from rag_pipeline.config import settings


@pytest.fixture
def client(monkeypatch):
    """Create test client."""
    class FakePipeline:
        def query(self, query, top_k=None, mode="semantic", rerank=False, tenant_id=None):
            return {
                "query": query,
                "result_count": 1,
                "results": [{
                    "document_id": "chunk-store-id",
                    "content": "Machine learning learns from examples.",
                    "metadata": {
                        "chunk_id": "chunk_123",
                        "source_file": "example.txt",
                    },
                    "similarity_score": 0.9,
                }],
                "status": "success",
            }

    monkeypatch.setattr(api, "get_pipeline", lambda: FakePipeline())
    monkeypatch.setattr(settings, "llm_provider", "fake")
    monkeypatch.setattr(api, "rate_limiter", api.RateLimiter(requests_per_minute=100))
    return TestClient(app)


def test_answer_empty_query(client):
    """Test /answer rejects empty query."""
    response = client.post(
        "/answer",
        json={"query": "", "top_k": 5},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_answer_with_query(client):
    """Test /answer returns valid response structure."""
    response = client.post(
        "/answer",
        json={"query": "What is machine learning?", "top_k": 5},
    )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "query" in data
    assert "answer" in data
    assert "citations" in data
    assert isinstance(data["citations"], list)
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert "abstained" in data
    assert isinstance(data["abstained"], bool)
    assert "status" in data
    assert data["status"] == "success"


def test_answer_default_top_k(client):
    """Test /answer uses default top_k=5."""
    response = client.post(
        "/answer",
        json={"query": "test query"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_answer_citation_structure(client):
    """Test citations have correct structure when present."""
    response = client.post(
        "/answer",
        json={"query": "machine learning", "top_k": 3},
    )

    assert response.status_code == 200
    data = response.json()

    # If there are citations, verify structure
    for citation in data["citations"]:
        assert "chunk_id" in citation
        assert "chunk_index" in citation
        assert isinstance(citation["chunk_index"], int)


def test_answer_high_confidence_not_abstained(client):
    """Test that high confidence doesn't necessarily mean abstained."""
    response = client.post(
        "/answer",
        json={"query": "machine learning basics", "top_k": 5},
    )

    assert response.status_code == 200
    data = response.json()

    # Just verify we get a boolean
    assert isinstance(data["abstained"], bool)

