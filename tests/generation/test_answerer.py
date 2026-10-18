"""Tests for answer generation and citation extraction."""

import pytest

from rag_pipeline.generation.answerer import (
    AnswerResult,
    Citation,
    generate_answer,
)
from rag_pipeline.generation.llm_client import FakeLLMClient


def test_citation_model():
    """Test Citation pydantic model."""
    citation = Citation(
        chunk_id="chunk_123",
        chunk_index=1,
        source_file="test.pdf",
        page_number=5,
    )
    assert citation.chunk_id == "chunk_123"
    assert citation.chunk_index == 1
    assert citation.source_file == "test.pdf"
    assert citation.page_number == 5


def test_answer_result_model():
    """Test AnswerResult pydantic model."""
    result = AnswerResult(
        answer="Test answer",
        citations=[],
        confidence=0.8,
        abstained=False,
    )
    assert result.answer == "Test answer"
    assert result.confidence == 0.8
    assert result.abstained is False


def test_answer_result_confidence_validation():
    """Test AnswerResult validates confidence is 0-1."""
    with pytest.raises(ValueError):
        AnswerResult(answer="test", confidence=1.5, citations=[])

    with pytest.raises(ValueError):
        AnswerResult(answer="test", confidence=-0.1, citations=[])


@pytest.mark.asyncio
async def test_generate_answer_empty_retrieval():
    """Test generate_answer handles empty retrieval."""
    client = FakeLLMClient()
    result = await generate_answer("query", [], client)

    assert result.abstained is True
    assert result.confidence == 0.0
    assert len(result.citations) == 0
    assert "context" in result.answer.lower()


@pytest.mark.asyncio
async def test_generate_answer_with_chunks():
    """Test generate_answer generates answer from chunks."""
    chunks = [
        {
            "content": "Machine learning is a field of AI.",
            "metadata": {
                "source_file": "ml.pdf",
                "page_number": 1,
                "chunk_id": "chunk_ml_1",
            },
            "score": 0.95,
        },
        {
            "content": "Deep learning uses neural networks.",
            "metadata": {
                "source_file": "dl.pdf",
                "page_number": 2,
                "chunk_id": "chunk_dl_1",
            },
            "score": 0.90,
        },
    ]
    client = FakeLLMClient(response="ML is AI. [chunk_ml_1]")
    result = await generate_answer("What is ML?", chunks, client)

    assert result.abstained is False
    assert result.confidence == (0.95 + 0.90) / 2
    assert "ML is AI" in result.answer


@pytest.mark.asyncio
async def test_generate_answer_extracts_citations_by_id():
    """Test citation extraction from chunk_id."""
    chunks = [
        {
            "content": "Context A",
            "metadata": {"chunk_id": "chunk_123", "source_file": "a.pdf"},
            "score": 0.9,
        },
        {
            "content": "Context B",
            "metadata": {"chunk_id": "chunk_456", "source_file": "b.pdf"},
            "score": 0.8,
        },
    ]
    client = FakeLLMClient(response="Answer based on [chunk_123] and [chunk_456].")
    result = await generate_answer("query", chunks, client)

    assert len(result.citations) == 2
    assert result.citations[0].chunk_id == "chunk_123"
    assert result.citations[0].source_file == "a.pdf"
    assert result.citations[1].chunk_id == "chunk_456"
    assert result.citations[1].source_file == "b.pdf"


@pytest.mark.asyncio
async def test_generate_answer_extracts_citations_by_index():
    """Test citation extraction by chunk index [1], [2], etc."""
    chunks = [
        {
            "content": "First chunk",
            "metadata": {"chunk_id": "c1", "source_file": "f.pdf"},
            "score": 0.9,
        },
        {
            "content": "Second chunk",
            "metadata": {"chunk_id": "c2", "source_file": "g.pdf"},
            "score": 0.8,
        },
    ]
    client = FakeLLMClient(response="Answer: [1] says something and [2] says another.")
    result = await generate_answer("query", chunks, client)

    assert len(result.citations) == 2
    assert result.citations[0].chunk_id == "c1"
    assert result.citations[1].chunk_id == "c2"


@pytest.mark.asyncio
async def test_generate_answer_deduplicates_citations():
    """Test that duplicate citations are only included once."""
    chunks = [
        {
            "content": "Content",
            "metadata": {"chunk_id": "chunk_1", "source_file": "f.pdf"},
            "score": 0.9,
        }
    ]
    client = FakeLLMClient(response="From [chunk_1] we know... [chunk_1] also shows...")
    result = await generate_answer("query", chunks, client)

    # Should only have one citation even though [chunk_1] appears twice
    assert len(result.citations) == 1


@pytest.mark.asyncio
async def test_generate_answer_ignores_invalid_citations():
    """Test that invalid citations are ignored."""
    chunks = [
        {
            "content": "Content",
            "metadata": {"chunk_id": "c1", "source_file": "f.pdf"},
            "score": 0.9,
        }
    ]
    # Reference out-of-range index and non-existent chunk_id
    client = FakeLLMClient(response="Based on [1] and [999] and [nonexistent]...")
    result = await generate_answer("query", chunks, client)

    # Should only extract [1], ignore [999] (out of range) and [nonexistent]
    assert len(result.citations) == 1
    assert result.citations[0].chunk_index == 1


@pytest.mark.asyncio
async def test_generate_answer_low_confidence_abstains():
    """Test that low confidence triggers abstention."""
    chunks = [
        {
            "content": "Content",
            "metadata": {"chunk_id": "c1"},
            "score": 0.2,  # Very low score
        }
    ]
    client = FakeLLMClient(response="Some answer")
    result = await generate_answer("query", chunks, client, abstain_threshold=0.5)

    assert result.abstained is True
    assert len(result.citations) == 0


@pytest.mark.asyncio
async def test_generate_answer_refusal_abstains():
    """Test that refusal phrases trigger abstention."""
    chunks = [
        {
            "content": "Content",
            "metadata": {"chunk_id": "c1"},
            "score": 0.9,  # High score, but will refuse
        }
    ]
    client = FakeLLMClient(response="I don't have enough context to answer this.")
    result = await generate_answer("query", chunks, client)

    assert result.abstained is True


@pytest.mark.asyncio
async def test_generate_answer_confidence_calculation():
    """Test confidence is mean of top-3 scores."""
    chunks = [
        {"content": "c1", "metadata": {"chunk_id": "c1"}, "score": 0.9},
        {"content": "c2", "metadata": {"chunk_id": "c2"}, "score": 0.8},
        {"content": "c3", "metadata": {"chunk_id": "c3"}, "score": 0.7},
        {"content": "c4", "metadata": {"chunk_id": "c4"}, "score": 0.1},
    ]
    client = FakeLLMClient(response="Answer")
    result = await generate_answer("query", chunks, client)

    expected = (0.9 + 0.8 + 0.7) / 3
    assert abs(result.confidence - expected) < 0.001
