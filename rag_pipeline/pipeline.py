"""Core RAG pipeline orchestrator."""

import hashlib
import uuid
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer

from rag_pipeline.chunking import ChunkingStrategyFactory
from rag_pipeline.config import settings
from rag_pipeline.loaders import DocumentLoaderFactory
from rag_pipeline.retrieval import get_reranker
from rag_pipeline.security.tenancy import get_current_tenant_id
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

    def ingest_document(
        self,
        file_path: Path,
        tenant_id: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> dict:
        """
        Complete document ingestion pipeline.

        Args:
            file_path: Path to document file

        Returns:
            dict with ingestion results
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        tenant_id = tenant_id or get_current_tenant_id()
        source_name = source_name or file_path.name
        doc_id = ""

        try:
            # A stable, tenant-scoped ID makes re-ingesting the same file idempotent.
            digest = hashlib.sha256()
            with file_path.open("rb") as source:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(block)
            file_hash = digest.hexdigest()
            doc_type = file_path.suffix.lstrip(".").lower()
            doc_id = str(
                uuid.uuid5(uuid.NAMESPACE_URL, f"rag:{tenant_id}:{doc_type}:{file_hash}")
            )

            # Step 1: Load document
            documents = DocumentLoaderFactory.load_document(file_path)
            if not documents:
                raise ValueError(f"No content extracted from {file_path}")

            for document in documents:
                document.metadata["source_file"] = source_name
                document.metadata["tenant_id"] = tenant_id
                # Do not expose temporary upload paths in indexed metadata.
                document.metadata["file_path"] = source_name

            # Step 2: Select and apply chunking strategy
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
                chunk.metadata["chunk_id"] = f"{doc_id}_{idx}"
                chunk.metadata["tenant_id"] = tenant_id

            # Step 4: Upsert to vector store
            upsert_result = self.vector_store.upsert(chunks)

            # Step 5: Register document
            self.document_registry[doc_id] = {
                "file_name": source_name,
                "file_path": source_name,
                "doc_type": doc_type,
                "chunk_count": len(chunks),
                "tenant_id": tenant_id,
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

    def query(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        mode: str = "semantic",
        rerank: bool = False,
        tenant_id: Optional[str] = None,
    ) -> dict:
        """
        Query the RAG pipeline for relevant chunks.

        Args:
            query_text: User query
            top_k: Number of top results (uses settings.top_k if None)
            mode: Search mode - 'semantic', 'keyword', or 'hybrid'
            rerank: Whether to apply cross-encoder reranking

        Returns:
            dict with search results
        """
        top_k = settings.top_k if top_k is None else top_k
        if top_k < 1:
            return {
                "query": query_text,
                "status": "error",
                "error": "top_k must be at least 1",
            }

        tenant_id = tenant_id or get_current_tenant_id()
        filters = {"tenant_id": tenant_id} if settings.api_keys else None
        candidate_k = max(top_k, settings.rerank_candidate_k) if rerank else top_k

        try:
            results = self.vector_store.search(
                query=query_text,
                top_k=candidate_k,
                min_score=settings.min_similarity_score,
                mode=mode,
                filters=filters,
            )

            # Convert to dicts for potential reranking
            result_dicts = [r.to_dict() for r in results]

            # Apply reranking if requested
            did_rerank = False
            if rerank and result_dicts:
                try:
                    reranker = get_reranker(
                        "cross-encoder",
                        embedding_model=self.embedding_model,
                        relevance_weight=settings.rerank_relevance_weight,
                    )
                    reranked_results = reranker.rerank(query_text, result_dicts, top_k=top_k)
                    result_dicts = reranked_results
                    did_rerank = True
                except Exception as e:
                    # Continue with the retrieval ranking if an optional model is unavailable.
                    print(f"Reranking failed: {e}")
                    result_dicts = result_dicts[:top_k]

            return {
                "query": query_text,
                "result_count": len(result_dicts),
                "results": result_dicts,
                "status": "success",
                "mode": mode,
                "reranked": did_rerank,
            }

        except Exception as e:
            return {
                "query": query_text,
                "status": "error",
                "error": str(e),
            }

    def delete_document(self, document_id: str, tenant_id: Optional[str] = None) -> dict:
        """Delete a document and its chunks from vector store."""
        try:
            tenant_id = tenant_id or get_current_tenant_id()
            scope = tenant_id if settings.api_keys else None
            result = self.vector_store.delete_by_document_id(document_id, tenant_id=scope)
            if document_id in self.document_registry:
                del self.document_registry[document_id]

            return {
                "document_id": document_id,
                "deleted_count": result.get("deleted_count", 0),
                "status": "deleted" if result.get("deleted_count", 0) else "not_found",
            }
        except Exception as e:
            return {
                "document_id": document_id,
                "status": "error",
                "error": str(e),
            }

    def clear_tenant(self, tenant_id: Optional[str] = None) -> dict:
        """Clear only the caller's tenant when API-key tenancy is enabled."""
        tenant_id = tenant_id or get_current_tenant_id()
        if settings.api_keys:
            result = self.vector_store.delete_by_filters({"tenant_id": tenant_id})
        else:
            result = self.vector_store.clear_all()
        if settings.api_keys:
            self.document_registry = {
                doc_id: info
                for doc_id, info in self.document_registry.items()
                if info.get("tenant_id") != tenant_id
            }
        else:
            self.document_registry.clear()
        return result

    def get_stats(self, tenant_id: Optional[str] = None) -> dict:
        """Get pipeline statistics."""
        tenant_id = tenant_id or get_current_tenant_id()
        filters = {"tenant_id": tenant_id} if settings.api_keys else None
        store_stats = self.vector_store.get_stats(filters=filters)
        if settings.api_keys:
            registered_documents = sum(
                1 for info in self.document_registry.values()
                if info.get("tenant_id") == tenant_id
            )
        else:
            registered_documents = len(self.document_registry)
        return {
            **store_stats,
            "registered_documents": registered_documents,
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
