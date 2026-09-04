"""Hybrid search combining BM25 keyword search with dense vector similarity."""

from typing import Optional
import re
import numpy as np


class BM25Search:
    """BM25 keyword search implementation."""

    def __init__(self, documents: list[dict]):
        """Initialize BM25 search with documents.

        Args:
            documents: List of dicts with 'id', 'content', and 'metadata'
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("rank-bm25 required for BM25 search: pip install rank-bm25")

        self.documents = documents
        self.doc_ids = [doc["id"] for doc in documents]

        # Tokenize documents
        tokenized_docs = [self._tokenize(doc["content"]) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization."""
        # Convert to lowercase and split on non-alphanumeric
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        """Search using BM25.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) tuples
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            (self.doc_ids[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]
        return results


class CrossEncoderReranker:
    """Cross-encoder based reranking."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError("sentence-transformers required for reranking")

        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Rerank documents using cross-encoder.

        Args:
            query: Search query
            documents: List of dicts with 'id' and 'content'
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) tuples, reranked
        """
        if not documents:
            return []

        # Create query-document pairs
        pairs = [
            [query, doc["content"]]
            for doc in documents
        ]

        # Get cross-encoder scores
        scores = self.model.predict(pairs)

        # Sort by score descending
        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        # Return top-k with doc IDs
        results = [
            (documents[idx]["id"], float(score))
            for idx, score in indexed_scores[:top_k]
        ]
        return results


class HybridSearcher:
    """Hybrid search combining semantic, keyword, and reranking."""

    def __init__(self, embedding_model, documents: Optional[list[dict]] = None):
        """Initialize hybrid searcher.

        Args:
            embedding_model: Sentence transformer model
            documents: Optional list of documents for BM25 indexing
        """
        self.embedding_model = embedding_model
        self.documents = documents or []
        self.bm25 = None
        self.reranker = None

        if self.documents:
            self._initialize_bm25()
            self._initialize_reranker()

    def _initialize_bm25(self):
        """Initialize BM25 search."""
        try:
            self.bm25 = BM25Search(self.documents)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize BM25: {e}")

    def _initialize_reranker(self):
        """Initialize cross-encoder reranker."""
        try:
            self.reranker = CrossEncoderReranker()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize reranker: {e}")

    def update_documents(self, documents: list[dict]):
        """Update documents for BM25 indexing."""
        self.documents = documents
        if self.documents:
            self._initialize_bm25()

    def _reciprocal_rank_fusion(
        self,
        semantic_results: list[tuple[str, float]],
        keyword_results: list[tuple[str, float]],
        k: int = 60,
    ) -> dict[str, float]:
        """Combine semantic and keyword results using reciprocal rank fusion.

        Args:
            semantic_results: List of (doc_id, score) from semantic search
            keyword_results: List of (doc_id, score) from BM25
            k: Constant for RRF (default 60)

        Returns:
            Dict of doc_id -> fused_score
        """
        fused_scores = {}

        # Add semantic search scores
        if semantic_results:
            for rank, (doc_id, _) in enumerate(semantic_results, 1):
                fused_scores[doc_id] = 1.0 / (k + rank)

        # Add keyword search scores
        if keyword_results:
            for rank, (doc_id, _) in enumerate(keyword_results, 1):
                if doc_id in fused_scores:
                    fused_scores[doc_id] += 1.0 / (k + rank)
                else:
                    fused_scores[doc_id] = 1.0 / (k + rank)

        return fused_scores

    def search(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
        mode: str = "hybrid",
    ) -> list[tuple[str, float]]:
        """Search using specified mode.

        Args:
            query: Search query
            documents: Documents to search over
            top_k: Number of results to return
            mode: 'semantic', 'keyword', or 'hybrid'

        Returns:
            List of (doc_id, score) tuples
        """
        if mode == "semantic":
            return self._semantic_search(query, documents, top_k)
        elif mode == "keyword":
            return self._keyword_search(query, top_k)
        elif mode == "hybrid":
            return self._hybrid_search(query, documents, top_k)
        else:
            raise ValueError(f"Unknown search mode: {mode}")

    def _semantic_search(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Semantic search only."""
        query_embedding = self.embedding_model.encode(query)

        scores = []
        for doc in documents:
            doc_embedding = self.embedding_model.encode(doc["content"])
            similarity = float(
                np.dot(query_embedding, doc_embedding) /
                (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-8)
            )
            scores.append((doc["id"], similarity))

        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _keyword_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Keyword search only using BM25."""
        if not self.bm25:
            raise RuntimeError("BM25 not initialized. Provide documents first.")

        results = self.bm25.search(query, top_k)
        return results

    def _hybrid_search(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Hybrid search with semantic + keyword + reranking."""
        # Step 1: Semantic search (top 20)
        semantic_results = self._semantic_search(query, documents, top_k=20)

        # Step 2: Keyword search with BM25 (top 20)
        try:
            bm25_temp = BM25Search(documents)
            keyword_results = bm25_temp.search(query, top_k=20)
        except Exception:
            keyword_results = []

        # Step 3: Reciprocal Rank Fusion
        if semantic_results or keyword_results:
            if semantic_results and keyword_results:
                fused_scores = self._reciprocal_rank_fusion(semantic_results, keyword_results)
            elif semantic_results:
                fused_scores = {doc_id: score for doc_id, score in semantic_results}
            else:
                fused_scores = {doc_id: score for doc_id, score in keyword_results}
        else:
            return []

        # Sort by fused score
        sorted_results = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        return sorted_results
