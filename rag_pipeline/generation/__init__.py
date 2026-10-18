"""Generation layer for RAG pipeline."""

from rag_pipeline.generation.answerer import (
    AnswerResult,
    Citation,
    generate_answer,
)
from rag_pipeline.generation.llm_client import (
    FakeLLMClient,
    GeminiClient,
    LLMClient,
    get_llm_client,
)

__all__ = [
    "LLMClient",
    "GeminiClient",
    "FakeLLMClient",
    "get_llm_client",
    "Citation",
    "AnswerResult",
    "generate_answer",
]
