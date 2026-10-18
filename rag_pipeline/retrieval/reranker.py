"""Reranker implementations for RAG retrieval."""

from typing import Any, Protocol


class Reranker(Protocol):
    """Protocol for rerankers."""

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """Rerank candidates based on query.

        Args:
            query: Query string.
            candidates: List of candidate dicts with 'content' and 'metadata'.
            top_k: Number of top results to return.

        Returns:
            Reranked list of candidates with updated scores.
        """
        ...


class NoOpReranker:
    """No-op reranker that returns candidates unchanged."""

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """Return top-k candidates without reranking.

        Args:
            query: Query string (unused).
            candidates: List of candidate dicts.
            top_k: Number of top results to return.

        Returns:
            Top-k candidates in original order.
        """
        return candidates[:top_k]


class CrossEncoderReranker:
    """Cross-encoder reranker using sentence-transformers."""

    def __init__(self, model: str = "BAAI/bge-reranker-base"):
        """Initialize cross-encoder reranker.

        Args:
            model: Model name from sentence-transformers.
        """
        from sentence_transformers import CrossEncoder

        self.model_name = model
        self.model = CrossEncoder(model)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """Rerank candidates using cross-encoder.

        Args:
            query: Query string.
            candidates: List of candidate dicts with 'content' and 'metadata'.
            top_k: Number of top results to return.

        Returns:
            Reranked list of top-k candidates with updated relevance scores.
        """
        if not candidates:
            return []

        # Extract candidate texts
        texts = [c.get("content", "") for c in candidates]

        # Get relevance scores
        scores = self.model.predict([[query, text] for text in texts])

        # Add scores to candidates and sort
        scored_candidates = []
        for i, candidate in enumerate(candidates):
            candidate_copy = candidate.copy()
            candidate_copy["relevance_score"] = float(scores[i])
            scored_candidates.append(candidate_copy)

        # Sort by relevance score (descending) and return top-k
        sorted_candidates = sorted(
            scored_candidates,
            key=lambda x: x.get("relevance_score", 0),
            reverse=True
        )
        return sorted_candidates[:top_k]


def get_reranker(name: str = "noop", **kwargs: Any) -> Any:
    """Factory function to create rerankers.

    Args:
        name: Reranker type ("noop" or "cross-encoder").
        **kwargs: Additional arguments passed to reranker constructor.

    Returns:
        Reranker instance.

    Raises:
        ValueError: If reranker type is unknown.
    """
    if name == "noop":
        return NoOpReranker()
    elif name == "cross-encoder":
        model = kwargs.get("model", "BAAI/bge-reranker-base")
        return CrossEncoderReranker(model=model)
    else:
        raise ValueError(f"Unknown reranker: {name}")
