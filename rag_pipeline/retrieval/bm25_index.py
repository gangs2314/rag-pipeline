"""BM25 index for keyword-based retrieval."""

from rank_bm25 import BM25Okapi


class BM25Index:
    """BM25 keyword search index."""

    def __init__(self):
        """Initialize BM25 index."""
        self.corpus = []
        self.index = None
        self.chunk_ids = []

    def build(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks.

        Args:
            chunks: List of chunk dicts with 'content' and 'metadata'.
        """
        self.corpus = chunks
        self.chunk_ids = []

        # Tokenize and build index
        tokenized_corpus = []
        for chunk in chunks:
            content = chunk.get("content", "").lower()
            tokens = content.split()
            tokenized_corpus.append(tokens)
            chunk_id = chunk.get("metadata", {}).get("chunk_id", f"chunk_{len(self.chunk_ids)}")
            self.chunk_ids.append(chunk_id)

        self.index = BM25Okapi(tokenized_corpus)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search using BM25.

        Args:
            query: Search query.
            top_k: Number of results to return.

        Returns:
            List of top-k chunks with BM25 scores.
        """
        if self.index is None or not self.corpus:
            return []

        # Tokenize query
        tokens = query.lower().split()

        # Get BM25 scores
        scores = self.index.get_scores(tokens)

        # Create result list with scores
        results = []
        for i, score in enumerate(scores):
            result = self.corpus[i].copy()
            result["bm25_score"] = float(score)
            results.append(result)

        # Sort by score and return top-k
        sorted_results = sorted(results, key=lambda x: x.get("bm25_score", 0), reverse=True)
        return sorted_results[:top_k]
