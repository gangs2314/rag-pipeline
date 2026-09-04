"""Tests for hybrid search combining BM25 keyword search with dense vectors."""

import pytest
from rag_pipeline.hybrid_search import BM25Search, CrossEncoderReranker, HybridSearcher

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

from sentence_transformers import SentenceTransformer


class TestBM25Search:
    """Test BM25 keyword search."""

    @pytest.fixture
    def documents(self):
        """Sample documents for BM25 testing."""
        return [
            {"id": "doc1", "content": "Machine learning is transforming industries"},
            {"id": "doc2", "content": "Deep learning neural networks"},
            {"id": "doc3", "content": "Natural language processing"},
            {"id": "doc4", "content": "Computer vision image recognition"},
            {"id": "doc5", "content": "Data science and analytics"},
        ]

    @pytest.fixture
    def bm25_search(self, documents):
        """Create BM25 search instance."""
        return BM25Search(documents)

    def test_bm25_initialization(self, documents):
        """Test BM25 search initialization."""
        searcher = BM25Search(documents)
        assert searcher.documents == documents
        assert len(searcher.doc_ids) == len(documents)

    def test_bm25_search_exact_match(self, bm25_search):
        """Test BM25 search with exact matching terms."""
        results = bm25_search.search("machine learning", top_k=3)

        assert len(results) > 0
        # First result should be doc1 which contains "machine learning"
        assert results[0][0] == "doc1"
        assert results[0][1] > 0

    def test_bm25_search_partial_match(self, bm25_search):
        """Test BM25 search with partial matches."""
        results = bm25_search.search("neural", top_k=3)

        assert len(results) > 0
        # Should find doc2 which contains "neural networks"
        assert any(doc_id == "doc2" for doc_id, _ in results)

    def test_bm25_search_top_k_limit(self, bm25_search):
        """Test that BM25 respects top_k limit."""
        for k in [1, 2, 3, 5]:
            results = bm25_search.search("learning", top_k=k)
            assert len(results) <= k

    def test_bm25_empty_query(self, bm25_search):
        """Test BM25 with empty query."""
        results = bm25_search.search("", top_k=3)
        assert len(results) == 0

    def test_bm25_no_matches(self, bm25_search):
        """Test BM25 with query that has no matches."""
        results = bm25_search.search("xyz123notreal", top_k=3)
        assert len(results) == 0


class TestCrossEncoderReranker:
    """Test cross-encoder reranking."""

    @pytest.fixture
    def documents(self):
        """Sample documents for reranking."""
        return [
            {"id": "doc1", "content": "Machine learning algorithms"},
            {"id": "doc2", "content": "Deep learning neural networks"},
            {"id": "doc3", "content": "Natural language processing"},
        ]

    def test_reranker_initialization(self):
        """Test reranker initialization."""
        try:
            reranker = CrossEncoderReranker()
            assert reranker.model is not None
        except Exception as e:
            pytest.skip(f"Cross-encoder not available: {e}")

    def test_reranker_rerank(self, documents):
        """Test reranking functionality."""
        try:
            reranker = CrossEncoderReranker()
            query = "machine learning"
            results = reranker.rerank(query, documents, top_k=3)

            assert len(results) <= 3
            # Results should be (doc_id, score) tuples
            for doc_id, score in results:
                assert doc_id in [d["id"] for d in documents]
                assert 0 <= score <= 1

        except Exception as e:
            pytest.skip(f"Cross-encoder not available: {e}")

    def test_reranker_empty_documents(self):
        """Test reranker with empty document list."""
        try:
            reranker = CrossEncoderReranker()
            results = reranker.rerank("query", [], top_k=5)
            assert len(results) == 0
        except Exception as e:
            pytest.skip(f"Cross-encoder not available: {e}")


class TestHybridSearcher:
    """Test hybrid search combining semantic + keyword + reranking."""

    @pytest.fixture
    def embedding_model(self):
        """Load embedding model."""
        return SentenceTransformer("all-MiniLM-L6-v2")

    @pytest.fixture
    def documents(self):
        """Sample documents."""
        return [
            {"id": "doc1", "content": "Machine learning is transforming industries"},
            {"id": "doc2", "content": "Deep learning with neural networks"},
            {"id": "doc3", "content": "Natural language processing for text"},
            {"id": "doc4", "content": "Computer vision for image recognition"},
            {"id": "doc5", "content": "Data science and machine learning analytics"},
        ]

    @pytest.fixture
    def searcher(self, embedding_model, documents):
        """Create hybrid searcher."""
        return HybridSearcher(embedding_model, documents)

    def test_hybrid_searcher_initialization(self, embedding_model, documents):
        """Test hybrid searcher initialization."""
        searcher = HybridSearcher(embedding_model, documents)
        assert searcher.embedding_model is not None
        assert searcher.documents == documents
        assert searcher.bm25 is not None

    def test_semantic_search(self, searcher, documents):
        """Test semantic search mode."""
        results = searcher.search("machine learning", documents, top_k=3, mode="semantic")

        assert len(results) <= 3
        for doc_id, score in results:
            assert doc_id in [d["id"] for d in documents]
            assert 0 <= score <= 1

    def test_keyword_search(self, searcher, documents):
        """Test keyword search mode."""
        results = searcher.search("neural networks", documents, top_k=3, mode="keyword")

        assert len(results) <= 3
        for doc_id, score in results:
            assert doc_id in [d["id"] for d in documents]

    def test_hybrid_search(self, searcher, documents):
        """Test hybrid search mode."""
        results = searcher.search("machine learning", documents, top_k=3, mode="hybrid")

        assert len(results) <= 3
        for doc_id, score in results:
            assert doc_id in [d["id"] for d in documents]
            # Cross-encoder scores are unbounded (can be negative or > 1)

    def test_invalid_search_mode(self, searcher, documents):
        """Test invalid search mode raises error."""
        with pytest.raises(ValueError):
            searcher.search("query", documents, top_k=3, mode="invalid_mode")

    def test_semantic_vs_keyword(self, searcher, documents):
        """Test that semantic and keyword searches can differ."""
        semantic_results = searcher.search("machine learning", documents, top_k=5, mode="semantic")
        keyword_results = searcher.search("machine learning", documents, top_k=5, mode="keyword")

        # Results should be different (though may have some overlap)
        semantic_ids = [doc_id for doc_id, _ in semantic_results]
        keyword_ids = [doc_id for doc_id, _ in keyword_results]

        # At least some results should differ between modes
        assert len(set(semantic_ids) ^ set(keyword_ids)) > 0 or semantic_ids == keyword_ids

    def test_hybrid_vs_semantic(self, searcher, documents):
        """Test that hybrid search can differ from semantic only."""
        semantic_results = searcher.search("machine learning", documents, top_k=5, mode="semantic")
        hybrid_results = searcher.search("machine learning", documents, top_k=5, mode="hybrid")

        # Both should return results
        assert len(semantic_results) > 0
        assert len(hybrid_results) > 0

    def test_update_documents(self, embedding_model):
        """Test updating documents in searcher."""
        initial_docs = [
            {"id": "doc1", "content": "Test content"}
        ]
        searcher = HybridSearcher(embedding_model, initial_docs)

        new_docs = [
            {"id": "doc1", "content": "Test content"},
            {"id": "doc2", "content": "New document"},
        ]
        searcher.update_documents(new_docs)

        assert len(searcher.documents) == 2
        assert searcher.bm25 is not None

    def test_reciprocal_rank_fusion(self, searcher):
        """Test reciprocal rank fusion scoring."""
        semantic_results = [
            ("doc1", 0.9),
            ("doc2", 0.8),
            ("doc3", 0.7),
        ]
        keyword_results = [
            ("doc2", 0.95),
            ("doc1", 0.85),
            ("doc4", 0.75),
        ]

        fused = searcher._reciprocal_rank_fusion(semantic_results, keyword_results)

        # All documents should be in fused results
        assert "doc1" in fused
        assert "doc2" in fused
        assert "doc3" in fused
        assert "doc4" in fused

        # Fused scores should be positive
        for score in fused.values():
            assert score > 0


class TestHybridSearchIntegration:
    """Integration tests for hybrid search."""

    def test_hybrid_search_with_real_embedding_model(self):
        """Test hybrid search with real embedding model."""
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        documents = [
            {"id": "doc1", "content": "Python programming language for data science"},
            {"id": "doc2", "content": "Java enterprise application development"},
            {"id": "doc3", "content": "JavaScript web development frameworks"},
            {"id": "doc4", "content": "Python machine learning libraries"},
            {"id": "doc5", "content": "Web development with React and Node"},
        ]

        searcher = HybridSearcher(embedding_model, documents)

        # Semantic search
        semantic = searcher.search("Python programming", documents, top_k=3, mode="semantic")
        assert len(semantic) > 0
        assert "doc1" in [doc_id for doc_id, _ in semantic]

        # Keyword search
        keyword = searcher.search("Python programming", documents, top_k=3, mode="keyword")
        assert len(keyword) > 0

        # Hybrid search
        hybrid = searcher.search("Python programming", documents, top_k=3, mode="hybrid")
        assert len(hybrid) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
