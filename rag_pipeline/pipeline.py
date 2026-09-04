"""Core RAG pipeline orchestrator."""

import uuid
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer
from rag_pipeline.config import settings
from rag_pipeline.loaders import DocumentLoaderFactory
from rag_pipeline.chunking import ChunkingStrategyFactory
from rag_pipeline.vector_store import VectorStoreFactory


class RAGPipeline:
    """Main RAG pipeline orchestrator."""

    def __init__(self):
        """Initialize the RAG pipeline."""
        self.embedding_model = self._load_embedding_model()
        self.vector_store = VectorStoreFactory.create(self.embedding_model)
        self.document_registry = {}  # Track uploaded documents

    def _load_embedding_model(self) -> SentenceTransformer:
        """Load embedding model with caching."""
        try:
            model = SentenceTransformer(
                settings.embedding_model,
                cache_folder=str(settings.embedding_cache_dir),
                device=settings.embedding_device,
            )
            print(f"Loaded embedding model: {settings.embedding_model}")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model: {e}")

    def ingest_document(self, file_path: Path) -> dict:
        """
        Complete document ingestion pipeline.

        Args:
            file_path: Path to document file

        Returns:
            dict with ingestion results
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Generate document ID
        doc_id = str(uuid.uuid4())

        try:
            # Step 1: Load document
            documents = DocumentLoaderFactory.load_document(file_path)
            if not documents:
                raise ValueError(f"No content extracted from {file_path}")

            # Step 2: Select and apply chunking strategy
            doc_type = file_path.suffix.lstrip(".").lower()
            chunking_strategy = ChunkingStrategyFactory.auto_select_strategy(
                doc_type,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )

            # Handle parent-child chunking if enabled
            if settings.use_parent_child_chunking:
                from rag_pipeline.chunking import ParentChildChunkingStrategy

                chunking_strategy = ParentChildChunkingStrategy(
                    child_chunk_size=settings.child_chunk_size,
                    child_overlap=settings.chunk_overlap,
                    parent_chunk_size=settings.parent_chunk_size,
                    parent_overlap=settings.chunk_overlap,
                )

            chunks = chunking_strategy.chunk(documents)

            # Step 3: Add document ID and unique chunk IDs to metadata
            for idx, chunk in enumerate(chunks):
                chunk.metadata["document_id"] = doc_id
                # Create unique ID combining doc_id and chunk index
                chunk.metadata["unique_chunk_id"] = f"{doc_id}_{idx}"

            # Step 4: Upsert to vector store
            upsert_result = self.vector_store.upsert(chunks)

            # Step 5: Register document
            self.document_registry[doc_id] = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "doc_type": doc_type,
                "chunk_count": len(chunks),
                "status": "ingested",
            }

            return {
                "document_id": doc_id,
                "file_name": file_path.name,
                "chunk_count": len(chunks),
                "upserted_count": upsert_result.get("upserted_count", 0),
                "duplicate_count": upsert_result.get("duplicate_count", 0),
                "status": "success",
            }

        except Exception as e:
            return {
                "document_id": doc_id,
                "file_name": file_path.name,
                "status": "error",
                "error": str(e),
            }

    def query(self, query_text: str, top_k: Optional[int] = None, mode: str = "semantic") -> dict:
        """
        Query the RAG pipeline for relevant chunks.

        Args:
            query_text: User query
            top_k: Number of top results (uses settings.top_k if None)
            mode: Search mode - 'semantic', 'keyword', or 'hybrid'

        Returns:
            dict with search results
        """
        top_k = top_k or settings.top_k

        try:
            results = self.vector_store.search(
                query=query_text,
                top_k=top_k,
                min_score=settings.min_similarity_score,
                mode=mode,
            )

            return {
                "query": query_text,
                "result_count": len(results),
                "results": [r.to_dict() for r in results],
                "status": "success",
                "mode": mode,
            }

        except Exception as e:
            return {
                "query": query_text,
                "status": "error",
                "error": str(e),
            }

    def delete_document(self, document_id: str) -> dict:
        """Delete a document and its chunks from vector store."""
        try:
            # This is a simplified version; in production you'd track document->chunk mappings
            if document_id in self.document_registry:
                del self.document_registry[document_id]

            return {
                "document_id": document_id,
                "status": "deleted",
            }
        except Exception as e:
            return {
                "document_id": document_id,
                "status": "error",
                "error": str(e),
            }

    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        store_stats = self.vector_store.get_stats()
        return {
            **store_stats,
            "registered_documents": len(self.document_registry),
            "embedding_model": settings.embedding_model,
            "chunking_config": {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "use_parent_child": settings.use_parent_child_chunking,
            },
        }


# Global pipeline instance
_pipeline_instance: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Get or create global pipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline()
    return _pipeline_instance
