"""Reranker implementations for RAG retrieval."""

from threading import Lock
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
        scores = self.model.predict(
            [[query, text] for text in texts],
            batch_size=32,
            show_progress_bar=False,
        )

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


class CrossEncoderMMRReranker:
    """Cross-encoder relevance ranking with semantic diversity selection.

    The cross-encoder scores query/document relevance. Maximal Marginal
    Relevance (MMR) then reduces near-duplicate results using the existing
    sentence embedding model, while preserving the most relevant result first.
    """

    def __init__(
        self,
        cross_encoder: CrossEncoderReranker,
        embedding_model: Any,
        relevance_weight: float = 0.75,
    ):
        if not 0.0 <= relevance_weight <= 1.0:
            raise ValueError("relevance_weight must be between 0 and 1")
        self.cross_encoder = cross_encoder
        self.embedding_model = embedding_model
        self.relevance_weight = relevance_weight

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        if not candidates or top_k <= 0:
            return []

        scored = self.cross_encoder.rerank(query, candidates, top_k=len(candidates))
        if len(scored) <= 1:
            return scored[:top_k]

        texts = [candidate.get("content", "") for candidate in scored]
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)

        # Cross-encoder outputs may be logits. Scale them for the MMR tradeoff.
        raw_scores = [candidate.get("relevance_score", 0.0) for candidate in scored]
        low, high = min(raw_scores), max(raw_scores)
        if high == low:
            relevance = [1.0] * len(raw_scores)
        else:
            relevance = [(score - low) / (high - low) for score in raw_scores]

        selected: list[int] = []
        selected_scores: list[float] = []
        remaining = set(range(len(scored)))
        while remaining and len(selected) < min(top_k, len(scored)):
            best_index = None
            best_score = float("-inf")
            for index in remaining:
                if not selected:
                    redundancy = 0.0
                else:
                    redundancy = max(
                        _cosine_similarity(embeddings[index], embeddings[chosen])
                        for chosen in selected
                    )
                mmr_score = (
                    self.relevance_weight * relevance[index]
                    - (1.0 - self.relevance_weight) * redundancy
                )
                if mmr_score > best_score:
                    best_index, best_score = index, mmr_score
            selected.append(best_index)
            selected_scores.append(best_score)
            remaining.remove(best_index)

        results = []
        for index, mmr_score in zip(selected, selected_scores):
            result = scored[index].copy()
            result["mmr_score"] = float(mmr_score)
            results.append(result)
        return results


def _cosine_similarity(left: Any, right: Any) -> float:
    """Cosine similarity for normalized vectors, without a NumPy dependency."""
    left_values = left.tolist() if hasattr(left, "tolist") else left
    right_values = right.tolist() if hasattr(right, "tolist") else right
    dot = sum(float(a) * float(b) for a, b in zip(left_values, right_values))
    left_norm = sum(float(a) ** 2 for a in left_values) ** 0.5
    right_norm = sum(float(b) ** 2 for b in right_values) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


_cross_encoder_cache: dict[str, CrossEncoderReranker] = {}
_cross_encoder_lock = Lock()


def _cached_cross_encoder(model: str) -> CrossEncoderReranker:
    with _cross_encoder_lock:
        if model not in _cross_encoder_cache:
            _cross_encoder_cache[model] = CrossEncoderReranker(model=model)
        return _cross_encoder_cache[model]


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
        reranker = _cached_cross_encoder(model)
        embedding_model = kwargs.get("embedding_model")
        if embedding_model is not None:
            return CrossEncoderMMRReranker(
                reranker,
                embedding_model,
                relevance_weight=kwargs.get("relevance_weight", 0.75),
            )
        return reranker
    else:
        raise ValueError(f"Unknown reranker: {name}")
