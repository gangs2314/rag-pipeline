"""Tests for prompt templates."""

from rag_pipeline.generation.prompt_templates import (
    ANSWER_PROMPT,
    SYSTEM_PROMPT,
    format_context_block,
)


def test_system_prompt_exists():
    """Test that SYSTEM_PROMPT is defined and contains key rules."""
    assert SYSTEM_PROMPT
    assert "context" in SYSTEM_PROMPT.lower()
    assert "cite" in SYSTEM_PROMPT.lower()
    assert "chunk_id" in SYSTEM_PROMPT


def test_answer_prompt_has_placeholders():
    """Test ANSWER_PROMPT has required format placeholders."""
    assert "{query}" in ANSWER_PROMPT
    assert "{context_block}" in ANSWER_PROMPT


def test_format_context_block_empty():
    """Test format_context_block handles empty list."""
    result = format_context_block([])
    assert result == ""


def test_format_context_block_single_chunk():
    """Test format_context_block formats a single chunk."""
    chunks = [
        {
            "content": "This is chunk content.",
            "metadata": {
                "source_file": "test.pdf",
                "page_number": 1,
                "chunk_id": "chunk_abc123",
            },
            "score": 0.95,
        }
    ]
    result = format_context_block(chunks)

    assert "[1]" in result
    assert "source=test.pdf" in result
    assert "page=1" in result
    assert "chunk_id=chunk_abc123" in result
    assert "This is chunk content." in result


def test_format_context_block_multiple_chunks():
    """Test format_context_block with multiple chunks."""
    chunks = [
        {
            "content": "Content 1",
            "metadata": {"source_file": "file1.pdf", "chunk_id": "id1"},
        },
        {
            "content": "Content 2",
            "metadata": {"source_file": "file2.pdf", "chunk_id": "id2"},
        },
    ]
    result = format_context_block(chunks)

    assert "[1]" in result
    assert "[2]" in result
    assert "Content 1" in result
    assert "Content 2" in result
    assert "source=file1.pdf" in result
    assert "source=file2.pdf" in result


def test_format_context_block_missing_metadata():
    """Test format_context_block handles missing metadata gracefully."""
    chunks = [
        {
            "content": "Content without metadata",
            "metadata": {},
            "score": 0.9,
        }
    ]
    result = format_context_block(chunks)

    assert "[1]" in result
    assert "source=unknown" in result
    assert "chunk_1" in result
    assert "Content without metadata" in result


def test_format_context_block_no_page_number():
    """Test format_context_block omits page number when not present."""
    chunks = [
        {
            "content": "Content",
            "metadata": {"source_file": "test.md", "chunk_id": "ch1"},
        }
    ]
    result = format_context_block(chunks)

    assert "source=test.md" in result
    assert "page=" not in result


def test_format_context_block_ordering():
    """Test format_context_block maintains chunk order."""
    chunks = [
        {"content": f"Chunk {i}", "metadata": {"chunk_id": f"id{i}"}}
        for i in range(1, 4)
    ]
    result = format_context_block(chunks)

    lines = result.split("\n")
    chunk_headers = [line for line in lines if line.startswith("[")]
    assert chunk_headers[0].startswith("[1]")
    assert chunk_headers[1].startswith("[2]")
    assert chunk_headers[2].startswith("[3]")
