"""Answer generation and citation extraction."""

import re
from typing import Any, Optional

from pydantic import BaseModel, Field

from rag_pipeline.generation.llm_client import LLMClient
from rag_pipeline.generation.prompt_templates import (
    ANSWER_PROMPT,
    SYSTEM_PROMPT,
    format_context_block,
)


class Citation(BaseModel):
    """A citation to a source chunk."""

    chunk_id: str = Field(..., description="Chunk ID referenced")
    chunk_index: int = Field(..., description="Position in retrieval results (1-indexed)")
    source_file: Optional[str] = Field(None, description="Source file name")
    page_number: Optional[int] = Field(None, description="Page number if available")


class AnswerResult(BaseModel):
    """Complete answer with citations and metadata."""

    answer: str = Field(..., description="Generated answer text")
    citations: list[Citation] = Field(default_factory=list, description="Citations in answer")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence (mean of top-3 retrieval scores)"
    )
    abstained: bool = Field(default=False, description="True if model refused to answer")


async def generate_answer(
    query: str,
    retrieved_chunks: list[dict],
    llm_client: LLMClient,
    abstain_threshold: float = 0.35,
) -> AnswerResult:
    """Generate an answer with citations from retrieved chunks.

    Args:
        query: User query.
        retrieved_chunks: List of chunk dicts with 'content', 'metadata', 'score'.
        llm_client: LLM client to use for generation.
        abstain_threshold: Confidence threshold below which to abstain.

    Returns:
        AnswerResult with answer, citations, confidence, and abstention flag.
    """
    # Handle empty retrieval
    if not retrieved_chunks:
        return AnswerResult(
            answer="I don't have any context to answer this question.",
            citations=[],
            confidence=0.0,
            abstained=True,
        )

    # Compute confidence from top-3 retrieval scores
    scores = [chunk.get("score", 0.0) for chunk in retrieved_chunks[:3]]
    confidence = sum(scores) / len(scores) if scores else 0.0

    # Format context block
    context_block = format_context_block(retrieved_chunks)

    # Generate answer
    prompt = ANSWER_PROMPT.format(query=query, context_block=context_block)
    answer_text = await llm_client.generate(SYSTEM_PROMPT, prompt)

    # Check for refusal patterns
    refusal_phrases = [
        "i don't have enough context",
        "insufficient context",
        "cannot answer",
        "not enough information",
    ]
    is_refusal = any(phrase in answer_text.lower() for phrase in refusal_phrases)

    # Determine abstention
    abstained = is_refusal or confidence < abstain_threshold

    if abstained:
        return AnswerResult(
            answer=answer_text,
            citations=[],
            confidence=confidence,
            abstained=True,
        )

    # Extract citations via regex: [chunk_id] or [N]
    citations_list = []
    citation_pattern = r"\[([a-zA-Z0-9_-]+)\]"
    matches = re.finditer(citation_pattern, answer_text)

    seen_citations = set()
    for match in matches:
        citation_text = match.group(1)
        chunk: dict[str, Any] | None = None
        chunk_index = 0
        chunk_id = ""

        # Try to parse as index number (1-based)
        try:
            chunk_index_int = int(citation_text)
            if 1 <= chunk_index_int <= len(retrieved_chunks):
                chunk = retrieved_chunks[chunk_index_int - 1]
                chunk_id = chunk.get("metadata", {}).get("chunk_id", citation_text)
                chunk_index = chunk_index_int
            else:
                continue
        except ValueError:
            # It's a chunk_id string
            chunk_id = citation_text
            # Find matching chunk
            for i, c in enumerate(retrieved_chunks):
                metadata = c.get("metadata")
                if metadata and metadata.get("chunk_id") == chunk_id:
                    chunk = c
                    chunk_index = i + 1
                    break
            if not chunk:
                continue

        if chunk_id not in seen_citations and chunk:
            seen_citations.add(chunk_id)
            metadata = chunk.get("metadata") or {}
            citations_list.append(
                Citation(
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    source_file=metadata.get("source_file"),
                    page_number=metadata.get("page_number"),
                )
            )

    return AnswerResult(
        answer=answer_text,
        citations=citations_list,
        confidence=confidence,
        abstained=False,
    )
