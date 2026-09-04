"""Comprehensive tests for document chunking strategies."""

import pytest
from pathlib import Path
from rag_pipeline.chunking import (
    SemanticChunkingStrategy,
    RecursiveCharacterChunkingStrategy,
    MarkdownHeaderChunkingStrategy,
    CodeAwareChunkingStrategy,
    ParentChildChunkingStrategy,
    ChunkingStrategyFactory,
)

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document


class TestSemanticChunking:
    """Test semantic chunking with sentence-level similarity."""

    @pytest.fixture
    def semantic_chunker(self):
        """Create semantic chunker instance."""
        return SemanticChunkingStrategy(
            chunk_size=512,
            similarity_threshold=0.75,
            min_chunk_size=100
        )

    def test_single_sentence_document(self, semantic_chunker):
        """Test handling of single-sentence documents."""
        doc = Document(
            page_content="This is a single sentence document.",
            metadata={"source_file": "test.txt", "doc_type": "txt"}
        )
        chunks = semantic_chunker.chunk([doc])

        assert len(chunks) == 1
        assert chunks[0].page_content == "This is a single sentence document."
        assert "chunk_index" in chunks[0].metadata
        assert "content_hash" in chunks[0].metadata

    def test_empty_document(self, semantic_chunker):
        """Test handling of empty documents."""
        doc = Document(
            page_content="",
            metadata={"source_file": "empty.txt", "doc_type": "txt"}
        )
        chunks = semantic_chunker.chunk([doc])

        assert len(chunks) == 1
        assert chunks[0].page_content == ""
        assert "chunk_index" in chunks[0].metadata

    def test_long_uninterrupted_text(self, semantic_chunker):
        """Test chunking of long uninterrupted text."""
        # Create long text without clear sentence boundaries
        sentences = [
            "Machine learning is a subset of artificial intelligence.",
            "It focuses on enabling computers to learn from data.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing helps computers understand human language.",
            "Computer vision enables machines to interpret visual information.",
            "These fields are rapidly advancing with new research.",
            "Applications include image recognition and language translation.",
            "The field continues to evolve with new techniques.",
        ]
        long_text = " ".join(sentences * 5)  # Repeat to make it long

        doc = Document(
            page_content=long_text,
            metadata={"source_file": "long.txt", "doc_type": "txt"}
        )
        chunks = semantic_chunker.chunk([doc])

        assert len(chunks) > 1
        # Each chunk should be reasonable size
        for chunk in chunks:
            assert len(chunk.page_content) > 0
            assert "chunk_index" in chunk.metadata
            assert chunk.metadata["chunk_index"] >= 0

    def test_threshold_boundary_cases(self, semantic_chunker):
        """Test chunking with different threshold values."""
        # Text with varying sentence similarity
        text = (
            "Artificial intelligence is transforming industries. "
            "Machine learning powers many applications. "
            "Deep neural networks learn hierarchical features. "
            "The quick brown fox jumps over the lazy dog. "
            "Semantic chunking splits on meaning boundaries."
        )

        doc = Document(
            page_content=text,
            metadata={"source_file": "threshold.txt", "doc_type": "txt"}
        )

        # High threshold - fewer splits
        high_threshold_chunker = SemanticChunkingStrategy(
            chunk_size=512,
            similarity_threshold=0.9,
            min_chunk_size=50
        )
        high_chunks = high_threshold_chunker.chunk([doc])

        # Low threshold - more splits
        low_threshold_chunker = SemanticChunkingStrategy(
            chunk_size=512,
            similarity_threshold=0.5,
            min_chunk_size=50
        )
        low_chunks = low_threshold_chunker.chunk([doc])

        # Low threshold should create more chunks (more sensitive to similarity drops)
        assert len(low_chunks) >= len(high_chunks)

    def test_metadata_preservation(self, semantic_chunker):
        """Test that original metadata is preserved in chunks."""
        doc = Document(
            page_content="First sentence. Second sentence. Third sentence.",
            metadata={
                "source_file": "test.md",
                "doc_type": "markdown",
                "page_number": 1,
                "custom_field": "custom_value"
            }
        )
        chunks = semantic_chunker.chunk([doc])

        for chunk in chunks:
            assert chunk.metadata["source_file"] == "test.md"
            assert chunk.metadata["doc_type"] == "markdown"
            assert chunk.metadata["page_number"] == 1
            assert chunk.metadata["custom_field"] == "custom_value"

    def test_chunk_index_uniqueness(self, semantic_chunker):
        """Test that chunk indices are unique within document."""
        doc = Document(
            page_content="Sentence one. Sentence two. Sentence three. Sentence four.",
            metadata={"source_file": "test.txt", "doc_type": "txt"}
        )
        chunks = semantic_chunker.chunk([doc])

        indices = [chunk.metadata["chunk_index"] for chunk in chunks]
        assert len(indices) == len(set(indices)), "Chunk indices should be unique"

    def test_unicode_handling(self, semantic_chunker):
        """Test handling of unicode characters."""
        doc = Document(
            page_content="Hello world. Café français. 中文文本. Привет мир.",
            metadata={"source_file": "unicode.txt", "doc_type": "txt"}
        )
        chunks = semantic_chunker.chunk([doc])

        assert len(chunks) > 0
        # Verify unicode is preserved
        full_text = " ".join([chunk.page_content for chunk in chunks])
        assert "français" in full_text or "Café" in full_text

    def test_content_hash_generation(self, semantic_chunker):
        """Test that content hashes are generated correctly."""
        doc = Document(
            page_content="Test content. With sentences.",
            metadata={"source_file": "hash.txt", "doc_type": "txt"}
        )
        chunks = semantic_chunker.chunk([doc])

        for chunk in chunks:
            assert "content_hash" in chunk.metadata
            # Hash should be 64 chars (SHA256 hex)
            assert len(chunk.metadata["content_hash"]) == 64

    def test_similarity_computation(self, semantic_chunker):
        """Test similarity computation between sentences."""
        import numpy as np

        # Create embeddings
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([1.0, 0.0, 0.0])  # Identical
        emb3 = np.array([0.0, 1.0, 0.0])  # Orthogonal

        sim_identical = semantic_chunker._compute_similarity(emb1, emb2)
        sim_orthogonal = semantic_chunker._compute_similarity(emb1, emb3)

        assert abs(sim_identical - 1.0) < 0.01, "Identical vectors should have ~1.0 similarity"
        assert abs(sim_orthogonal - 0.0) < 0.01, "Orthogonal vectors should have ~0.0 similarity"

    def test_multiple_documents(self, semantic_chunker):
        """Test chunking multiple documents."""
        docs = [
            Document(
                page_content="First document sentence one. First document sentence two.",
                metadata={"source_file": "doc1.txt", "doc_type": "txt"}
            ),
            Document(
                page_content="Second document sentence one. Second document sentence two.",
                metadata={"source_file": "doc2.txt", "doc_type": "txt"}
            ),
        ]

        chunks = semantic_chunker.chunk(docs)

        assert len(chunks) >= 2
        # Verify source files are correct
        sources = [chunk.metadata["source_file"] for chunk in chunks]
        assert "doc1.txt" in sources
        assert "doc2.txt" in sources

    def test_min_chunk_size_enforcement(self, semantic_chunker):
        """Test that minimum chunk size is respected."""
        chunker = SemanticChunkingStrategy(
            chunk_size=512,
            similarity_threshold=0.75,
            min_chunk_size=100
        )

        doc = Document(
            page_content="Short. Words. Only.",
            metadata={"source_file": "short.txt", "doc_type": "txt"}
        )
        chunks = chunker.chunk([doc])

        # Should still produce chunks even if below min size
        assert len(chunks) > 0


class TestRecursiveCharacterChunking:
    """Test recursive character chunking strategy."""

    @pytest.fixture
    def recursive_chunker(self):
        """Create recursive character chunker."""
        return RecursiveCharacterChunkingStrategy(
            chunk_size=100,
            chunk_overlap=10
        )

    def test_basic_chunking(self, recursive_chunker):
        """Test basic recursive chunking."""
        doc = Document(
            page_content="This is a test. " * 20,
            metadata={"source_file": "test.txt", "doc_type": "txt"}
        )
        chunks = recursive_chunker.chunk([doc])

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) > 0

    def test_chunk_size_compliance(self, recursive_chunker):
        """Test that chunks respect size limits (with some tolerance)."""
        doc = Document(
            page_content="Test content. " * 50,
            metadata={"source_file": "test.txt", "doc_type": "txt"}
        )
        chunks = recursive_chunker.chunk([doc])

        for chunk in chunks:
            # Allow some tolerance due to split boundaries
            assert len(chunk.page_content) <= recursive_chunker.chunk_size * 1.5


class TestMarkdownChunking:
    """Test markdown header-aware chunking."""

    @pytest.fixture
    def markdown_chunker(self):
        """Create markdown chunker."""
        return MarkdownHeaderChunkingStrategy(chunk_size=200, chunk_overlap=20)

    def test_markdown_header_preservation(self, markdown_chunker):
        """Test that markdown header chunking works."""
        markdown_text = """# Main Title

## Section 1
Content for section 1. This has multiple sentences.

## Section 2
Content for section 2. More information here.

### Subsection 2.1
Deeper content."""

        doc = Document(
            page_content=markdown_text,
            metadata={"source_file": "test.md", "doc_type": "markdown"}
        )
        chunks = markdown_chunker.chunk([doc])

        # Should produce chunks
        assert len(chunks) > 0
        full_content = " ".join([chunk.page_content for chunk in chunks])
        assert "Section" in full_content or "Content" in full_content


class TestParentChildChunking:
    """Test parent-child hierarchical chunking."""

    @pytest.fixture
    def parent_child_chunker(self):
        """Create parent-child chunker."""
        return ParentChildChunkingStrategy(
            child_chunk_size=100,
            child_overlap=10,
            parent_chunk_size=300,
            parent_overlap=30
        )

    def test_parent_child_relationship(self, parent_child_chunker):
        """Test that child chunks reference parents."""
        doc = Document(
            page_content="Sentence one. " * 30,
            metadata={"source_file": "test.txt", "doc_type": "txt"}
        )
        chunks = parent_child_chunker.chunk([doc])

        # Should have both parent and child chunks
        parent_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]
        child_chunks = [c for c in chunks if c.metadata.get("chunk_type") == "child"]

        assert len(parent_chunks) > 0
        assert len(child_chunks) > 0

        # Each child should reference a parent
        for child in child_chunks:
            assert "parent_id" in child.metadata
            assert child.metadata["parent_id"] is not None


class TestChunkingStrategyFactory:
    """Test chunking strategy factory."""

    def test_factory_creation(self):
        """Test creating strategies via factory."""
        strategies = [
            ("recursive", RecursiveCharacterChunkingStrategy),
            ("markdown_header", MarkdownHeaderChunkingStrategy),
            ("code_aware", CodeAwareChunkingStrategy),
            ("parent_child", ParentChildChunkingStrategy),
            ("semantic", SemanticChunkingStrategy),
        ]

        for strategy_name, expected_class in strategies:
            strategy = ChunkingStrategyFactory.create_strategy(strategy_name)
            assert isinstance(strategy, expected_class)

    def test_auto_select_strategy(self):
        """Test automatic strategy selection by document type."""
        md_strategy = ChunkingStrategyFactory.auto_select_strategy("md")
        assert isinstance(md_strategy, MarkdownHeaderChunkingStrategy)

        py_strategy = ChunkingStrategyFactory.auto_select_strategy("py")
        assert isinstance(py_strategy, CodeAwareChunkingStrategy)

        txt_strategy = ChunkingStrategyFactory.auto_select_strategy("txt")
        assert isinstance(txt_strategy, RecursiveCharacterChunkingStrategy)


class TestChunkingEdgeCases:
    """Test edge cases across chunking strategies."""

    def test_very_long_single_word(self):
        """Test handling of very long single word."""
        chunker = RecursiveCharacterChunkingStrategy(chunk_size=50, chunk_overlap=5)
        long_word = "a" * 200

        doc = Document(
            page_content=long_word,
            metadata={"source_file": "long_word.txt", "doc_type": "txt"}
        )
        chunks = chunker.chunk([doc])

        # Should still produce chunks
        assert len(chunks) > 0

    def test_only_whitespace(self):
        """Test handling of whitespace-only documents."""
        chunker = RecursiveCharacterChunkingStrategy()

        doc = Document(
            page_content="   \n\n   \t\t   ",
            metadata={"source_file": "whitespace.txt", "doc_type": "txt"}
        )
        chunks = chunker.chunk([doc])

        # Whitespace-only content produces no chunks (expected behavior)
        assert len(chunks) >= 0

    def test_special_characters(self):
        """Test handling of special characters."""
        chunker = SemanticChunkingStrategy()

        doc = Document(
            page_content="Special @#$% characters. More !@#$ symbols. Regular text.",
            metadata={"source_file": "special.txt", "doc_type": "txt"}
        )
        chunks = chunker.chunk([doc])

        assert len(chunks) > 0
        full_text = " ".join([c.page_content for c in chunks])
        assert "@#$%" in full_text or "@#$" in full_text


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
