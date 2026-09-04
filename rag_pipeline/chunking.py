"""Advanced chunking strategies for document processing."""

import hashlib
from abc import ABC, abstractmethod
from typing import Optional
from datetime import datetime

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

try:
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter,
        MarkdownHeaderTextSplitter,
    )
except ImportError:
    from langchain.text_splitter import (
        RecursiveCharacterTextSplitter,
        MarkdownHeaderTextSplitter,
    )


class BaseChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Initialize chunking strategy."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents into chunks."""
        pass

    def _compute_content_hash(self, content: str) -> str:
        """Compute SHA256 hash of content for deduplication."""
        return hashlib.sha256(content.encode()).hexdigest()

    def _add_chunk_metadata(
        self,
        chunk: Document,
        chunk_index: int,
        parent_id: Optional[str] = None,
        **kwargs
    ) -> Document:
        """Add standard chunking metadata to document."""
        chunk.metadata["chunk_index"] = chunk_index
        chunk.metadata["chunk_created_at"] = datetime.utcnow().isoformat()
        chunk.metadata["content_hash"] = self._compute_content_hash(chunk.page_content)

        if parent_id:
            chunk.metadata["parent_id"] = parent_id

        chunk.metadata.update(kwargs)
        return chunk


class RecursiveCharacterChunkingStrategy(BaseChunkingStrategy):
    """Default recursive character splitting strategy."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        separators: Optional[list[str]] = None,
    ):
        """Initialize with custom separators."""
        super().__init__(chunk_size, chunk_overlap)
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents using recursive character splitting."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

        chunked_docs = []
        for doc in documents:
            splits = splitter.split_documents([doc])
            for chunk_idx, split in enumerate(splits):
                self._add_chunk_metadata(split, chunk_idx)
                chunked_docs.append(split)

        return chunked_docs


class MarkdownHeaderChunkingStrategy(BaseChunkingStrategy):
    """Header-aware chunking for Markdown and structured documents."""

    HEADER_SEPARATORS = [
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
        ("####", "H4"),
    ]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Initialize with Markdown header awareness."""
        super().__init__(chunk_size, chunk_overlap)

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split by headers first, then recursively split within sections."""
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.HEADER_SEPARATORS,
            return_each_line=False,
        )

        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunked_docs = []
        for doc in documents:
            # For markdown documents
            if doc.metadata.get("doc_type") in ["md", "markdown"]:
                try:
                    header_splits = markdown_splitter.split_text(doc.page_content)
                    for split in header_splits:
                        split.metadata.update(doc.metadata)
                        # Further split large header sections
                        if len(split.page_content) > self.chunk_size * 2:
                            sub_splits = recursive_splitter.split_documents([split])
                            for chunk_idx, sub_split in enumerate(sub_splits):
                                self._add_chunk_metadata(sub_split, chunk_idx)
                                chunked_docs.append(sub_split)
                        else:
                            self._add_chunk_metadata(split, 0)
                            chunked_docs.append(split)
                except Exception:
                    # Fallback to recursive splitting
                    splits = recursive_splitter.split_documents([doc])
                    for chunk_idx, split in enumerate(splits):
                        self._add_chunk_metadata(split, chunk_idx)
                        chunked_docs.append(split)
            else:
                # For non-markdown, use recursive splitting
                splits = recursive_splitter.split_documents([doc])
                for chunk_idx, split in enumerate(splits):
                    self._add_chunk_metadata(split, chunk_idx)
                    chunked_docs.append(split)

        return chunked_docs


class CodeAwareChunkingStrategy(BaseChunkingStrategy):
    """Language-aware chunking for code files."""

    LANGUAGE_SEPARATORS = {
        "python": ["\nclass ", "\ndef ", "\n    def ", "\n\n", "\n", ""],
        "javascript": ["\nclass ", "\nfunction ", "\nconst ", "\nlet ", "\n\n", "\n", ""],
        "java": ["\nclass ", "\npublic ", "\nprivate ", "\nprotected ", "\n\n", "\n", ""],
        "cpp": ["\nclass ", "\nvoid ", "\nint ", "\nbool ", "\n\n", "\n", ""],
        "go": ["\nfunc ", "\ntype ", "\n\n", "\n", ""],
        "rust": ["\nfn ", "\npub fn ", "\nimpl ", "\n\n", "\n", ""],
        "typescript": ["\nclass ", "\nfunction ", "\nconst ", "\nexport ", "\n\n", "\n", ""],
    }

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split code files with language-specific separators."""
        chunked_docs = []

        for doc in documents:
            language = doc.metadata.get("language", "unknown").lower()
            separators = self.LANGUAGE_SEPARATORS.get(language, ["\n\n", "\n", ""])

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=separators,
            )

            splits = splitter.split_documents([doc])
            for chunk_idx, split in enumerate(splits):
                self._add_chunk_metadata(split, chunk_idx)
                chunked_docs.append(split)

        return chunked_docs


class ParentChildChunkingStrategy(BaseChunkingStrategy):
    """Hierarchical chunking: small child chunks for search, large parent chunks for context."""

    def __init__(
        self,
        child_chunk_size: int = 256,
        child_overlap: int = 50,
        parent_chunk_size: int = 1024,
        parent_overlap: int = 100,
    ):
        """Initialize parent-child chunking."""
        super().__init__(child_chunk_size, child_overlap)
        self.parent_chunk_size = parent_chunk_size
        self.parent_overlap = parent_overlap
        self.parent_id_counter = 0

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Create parent chunks first, then child chunks with parent references."""
        # First, create parent chunks
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        # Then create child chunks
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        all_chunks = []

        for doc in documents:
            parent_splits = parent_splitter.split_documents([doc])

            for parent_idx, parent_chunk in enumerate(parent_splits):
                parent_id = f"{doc.metadata['source_file']}_parent_{parent_idx}"
                parent_chunk.metadata["parent_id"] = None  # Parent has no parent
                parent_chunk.metadata["chunk_type"] = "parent"
                self._add_chunk_metadata(parent_chunk, parent_idx)
                all_chunks.append(parent_chunk)

                # Create child chunks from parent
                child_splits = child_splitter.split_documents([parent_chunk])
                for child_idx, child_chunk in enumerate(child_splits):
                    child_chunk.metadata["parent_id"] = parent_id
                    child_chunk.metadata["chunk_type"] = "child"
                    child_chunk.metadata["parent_content"] = parent_chunk.page_content
                    self._add_chunk_metadata(child_chunk, child_idx)
                    all_chunks.append(child_chunk)

        return all_chunks


class SemanticChunkingStrategy(BaseChunkingStrategy):
    """Semantic chunking based on sentence similarity."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.75,
        min_chunk_size: int = 100,
    ):
        """Initialize semantic chunking.

        Args:
            chunk_size: Target chunk size in characters (soft limit)
            chunk_overlap: Overlap between chunks (not used in semantic chunking)
            similarity_threshold: Cosine similarity threshold for sentence breaks (0-1)
            min_chunk_size: Minimum characters per chunk
        """
        super().__init__(chunk_size, chunk_overlap)
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self._embedding_model = None

    def _get_embedding_model(self):
        """Lazy load embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                raise RuntimeError(f"Failed to load embedding model for semantic chunking: {e}")
        return self._embedding_model

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _compute_similarity(self, embedding1, embedding2) -> float:
        """Compute cosine similarity between two embeddings."""
        import numpy as np
        return float(np.dot(embedding1, embedding2) /
                     (np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-8))

    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents using semantic similarity between consecutive sentences."""
        model = self._get_embedding_model()
        chunked_docs = []

        for doc in documents:
            sentences = self._split_into_sentences(doc.page_content)

            if not sentences:
                # Empty document
                doc.metadata["chunk_index"] = 0
                self._add_chunk_metadata(doc, 0)
                chunked_docs.append(doc)
                continue

            if len(sentences) == 1:
                # Single sentence
                doc.metadata["chunk_index"] = 0
                self._add_chunk_metadata(doc, 0)
                chunked_docs.append(doc)
                continue

            # Embed all sentences
            embeddings = model.encode(sentences, convert_to_numpy=True)

            # Find split points based on similarity drops
            chunks_text = []
            current_chunk = [sentences[0]]
            current_size = len(sentences[0])

            for i in range(1, len(sentences)):
                # Compute similarity between consecutive sentences
                similarity = self._compute_similarity(embeddings[i - 1], embeddings[i])

                # Decide whether to split
                should_split = (
                    similarity < self.similarity_threshold or
                    current_size + len(sentences[i]) > self.chunk_size * 1.5
                )

                if should_split and current_chunk and current_size >= self.min_chunk_size:
                    chunks_text.append(" ".join(current_chunk))
                    current_chunk = [sentences[i]]
                    current_size = len(sentences[i])
                else:
                    current_chunk.append(sentences[i])
                    current_size += len(sentences[i]) + 1  # +1 for space

            # Add remaining chunk
            if current_chunk:
                chunks_text.append(" ".join(current_chunk))

            # Create chunk documents
            for chunk_idx, chunk_text in enumerate(chunks_text):
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata=doc.metadata.copy()
                )
                self._add_chunk_metadata(chunk_doc, chunk_idx)
                chunked_docs.append(chunk_doc)

        return chunked_docs


class ChunkingStrategyFactory:
    """Factory for selecting and creating chunking strategies."""

    STRATEGIES = {
        "recursive": RecursiveCharacterChunkingStrategy,
        "markdown_header": MarkdownHeaderChunkingStrategy,
        "code_aware": CodeAwareChunkingStrategy,
        "parent_child": ParentChildChunkingStrategy,
        "semantic": SemanticChunkingStrategy,
    }

    @classmethod
    def create_strategy(
        cls,
        strategy_type: str = "recursive",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        **kwargs
    ) -> BaseChunkingStrategy:
        """Create chunking strategy instance."""
        if strategy_type not in cls.STRATEGIES:
            raise ValueError(f"Unknown chunking strategy: {strategy_type}")

        strategy_class = cls.STRATEGIES[strategy_type]
        return strategy_class(chunk_size, chunk_overlap, **kwargs)

    @classmethod
    def auto_select_strategy(cls, doc_type: str, **kwargs) -> BaseChunkingStrategy:
        """Auto-select best strategy based on document type."""
        if doc_type in ["md", "markdown"]:
            return cls.create_strategy("markdown_header", **kwargs)
        elif doc_type in ["py", "js", "ts", "java", "cpp", "go", "rs"]:
            return cls.create_strategy("code_aware", **kwargs)
        else:
            return cls.create_strategy("recursive", **kwargs)
