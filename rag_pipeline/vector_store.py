"""Vector store abstraction layer supporting ChromaDB and Qdrant."""

from abc import ABC, abstractmethod
from typing import Optional
import hashlib

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

from rag_pipeline.config import settings
from rag_pipeline.deduplication import get_dedup_tracker


class VectorStoreResult:
    """Result from vector store operations."""

    def __init__(
        self,
        document_id: str,
        content: str,
        metadata: dict,
        similarity_score: float = 0.0,
    ):
        self.document_id = document_id
        self.content = content
        self.metadata = metadata
        self.similarity_score = similarity_score

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "document_id": self.document_id,
            "content": self.content,
            "metadata": self.metadata,
            "similarity_score": self.similarity_score,
        }


class BaseVectorStore(ABC):
    """Abstract base class for vector stores."""

    def __init__(self, embedding_model, collection_name: str = "rag_documents"):
        """Initialize vector store."""
        self.embedding_model = embedding_model
        self.collection_name = collection_name

    @abstractmethod
    def upsert(self, documents: list[Document]) -> dict:
        """Upsert documents into vector store."""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filters: Optional[dict] = None,
        mode: str = "semantic",
    ) -> list[VectorStoreResult]:
        """Search for relevant documents.

        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            filters: Optional metadata filters
            mode: Search mode - 'semantic', 'keyword', or 'hybrid'
        """
        pass

    @abstractmethod
    def delete(self, document_ids: list[str]) -> dict:
        """Delete documents from vector store."""
        pass

    @abstractmethod
    def get_stats(self) -> dict:
        """Get vector store statistics."""
        pass

    @abstractmethod
    def clear_all(self) -> dict:
        """Clear all documents from vector store."""
        pass

    def _deduplicate_documents(
        self,
        documents: list[Document],
    ) -> tuple[list[Document], int]:
        """Remove duplicate documents by content hash and normalize whitespace."""
        seen_hashes = set()
        unique_docs = []
        duplicate_count = 0

        for doc in documents:
            # Normalize content: strip whitespace, convert to lowercase for comparison
            normalized = ' '.join(doc.page_content.split()).lower()
            content_hash = hashlib.sha256(normalized.encode()).hexdigest()
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append(doc)
            else:
                duplicate_count += 1

        return unique_docs, duplicate_count


class ChromaDBVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation."""

    def __init__(self, embedding_model, collection_name: str = "rag_documents"):
        """Initialize ChromaDB vector store."""
        super().__init__(embedding_model, collection_name)
        try:
            import chromadb
            self.chroma_client = chromadb.PersistentClient(
                path=str(settings.chromadb_path)
            )
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB: {e}")

    def upsert(self, documents: list[Document]) -> dict:
        """Upsert documents into ChromaDB with advanced deduplication."""
        dedup_tracker = get_dedup_tracker()

        # Filter out duplicates using advanced deduplication
        unique_docs = []
        duplicate_count = 0

        for doc in documents:
            content = doc.page_content
            if dedup_tracker.add_content(content, doc.metadata.get('unique_chunk_id', '')):
                unique_docs.append(doc)
            else:
                duplicate_count += 1

        if not unique_docs:
            return {
                "status": "success",
                "upserted_count": 0,
                "duplicate_count": duplicate_count,
            }

        # Prepare data for upsert
        ids = []
        documents_content = []
        metadatas = []
        embeddings = []

        # Track what's already in the collection to avoid re-inserting
        existing_data = self.collection.get()
        existing_ids = set(existing_data["ids"]) if existing_data and existing_data.get("ids") else set()
        skipped_count = 0

        for doc in unique_docs:
            # Use unique_chunk_id if available
            doc_id = doc.metadata.get('unique_chunk_id')
            if not doc_id:
                doc_id = f"{doc.metadata.get('source_file', 'unknown')}_{doc.metadata.get('chunk_index', 0)}"

            # Skip if already exists in ChromaDB
            if doc_id in existing_ids:
                skipped_count += 1
                continue

            embedding = self.embedding_model.encode(doc.page_content).tolist()

            ids.append(doc_id)
            documents_content.append(doc.page_content)
            metadatas.append(doc.metadata)
            embeddings.append(embedding)

        if not ids:
            return {
                "status": "success",
                "upserted_count": 0,
                "duplicate_count": duplicate_count + skipped_count,
            }

        try:
            self.collection.upsert(
                ids=ids,
                documents=documents_content,
                metadatas=metadatas,
                embeddings=embeddings,
            )
            return {
                "status": "success",
                "upserted_count": len(ids),
                "duplicate_count": duplicate_count + skipped_count,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to upsert documents to ChromaDB: {e}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filters: Optional[dict] = None,
        mode: str = "semantic",
    ) -> list[VectorStoreResult]:
        """Search ChromaDB for relevant documents with hybrid search support.

        Args:
            query: Search query
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            filters: Optional metadata filters
            mode: Search mode - 'semantic', 'keyword', or 'hybrid'
        """
        try:
            if mode == "hybrid":
                return self._hybrid_search(query, top_k, min_score, filters)
            else:
                # Fall back to semantic search for non-hybrid modes
                return self._semantic_search(query, top_k, min_score, filters)
        except Exception as e:
            raise RuntimeError(f"Failed to search ChromaDB: {e}")

    def _semantic_search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filters: Optional[dict] = None,
    ) -> list[VectorStoreResult]:
        """Semantic search using vector similarity."""
        query_embedding = self.embedding_model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filters,
        )

        search_results = []
        if results["documents"] and len(results["documents"]) > 0:
            for i, doc_id in enumerate(results["ids"][0]):
                if results["distances"] and len(results["distances"]) > 0:
                    distance = results["distances"][0][i]
                    similarity = 1 - distance
                else:
                    similarity = 0.0

                if similarity >= min_score:
                    result = VectorStoreResult(
                        document_id=doc_id,
                        content=results["documents"][0][i],
                        metadata=results["metadatas"][0][i],
                        similarity_score=similarity,
                    )
                    search_results.append(result)

        return search_results

    def _hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filters: Optional[dict] = None,
    ) -> list[VectorStoreResult]:
        """Hybrid search using semantic + keyword + reranking."""
        from rag_pipeline.hybrid_search import HybridSearcher

        # Get all documents for hybrid search
        try:
            all_docs_data = self.collection.get()
        except Exception:
            return self._semantic_search(query, top_k, min_score, filters)

        if not all_docs_data.get("ids"):
            return []

        # Convert to hybrid searcher format
        documents = []
        for i, doc_id in enumerate(all_docs_data["ids"]):
            documents.append({
                "id": doc_id,
                "content": all_docs_data["documents"][i] if all_docs_data["documents"] else "",
                "metadata": all_docs_data["metadatas"][i] if all_docs_data["metadatas"] else {},
            })

        try:
            searcher = HybridSearcher(self.embedding_model, documents)
            hybrid_results = searcher.search(query, documents, top_k, mode="hybrid")
        except Exception:
            # Fall back to semantic search if hybrid fails
            return self._semantic_search(query, top_k, min_score, filters)

        # Convert hybrid results to VectorStoreResult
        search_results = []
        for doc_id, score in hybrid_results:
            if score >= min_score:
                # Find the document
                doc_index = all_docs_data["ids"].index(doc_id)
                result = VectorStoreResult(
                    document_id=doc_id,
                    content=all_docs_data["documents"][doc_index] if all_docs_data["documents"] else "",
                    metadata=all_docs_data["metadatas"][doc_index] if all_docs_data["metadatas"] else {},
                    similarity_score=score,
                )
                search_results.append(result)

        return search_results

    def delete(self, document_ids: list[str]) -> dict:
        """Delete documents from ChromaDB."""
        try:
            self.collection.delete(ids=document_ids)
            return {
                "status": "success",
                "deleted_count": len(document_ids),
            }
        except Exception as e:
            raise RuntimeError(f"Failed to delete from ChromaDB: {e}")

    def get_stats(self) -> dict:
        """Get ChromaDB collection statistics."""
        try:
            count = self.collection.count()
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "vector_store_type": "chromadb",
                "path": str(settings.chromadb_path),
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get ChromaDB stats: {e}")

    def clear_all(self) -> dict:
        """Clear all documents from ChromaDB collection."""
        try:
            # Get all document IDs
            all_docs = self.collection.get()
            cleared_count = 0
            if all_docs and all_docs.get("ids"):
                self.collection.delete(ids=all_docs["ids"])
                cleared_count = len(all_docs["ids"])

            # Reset deduplication tracker
            dedup_tracker = get_dedup_tracker()
            dedup_tracker.clear()

            return {
                "status": "success",
                "cleared_count": cleared_count,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to clear ChromaDB: {e}")


class QdrantVectorStore(BaseVectorStore):
    """Qdrant vector store implementation."""

    def __init__(self, embedding_model, collection_name: str = "rag_documents"):
        """Initialize Qdrant vector store."""
        super().__init__(embedding_model, collection_name)
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
            )

            # Get embedding dimension from model
            embedding_dim = len(embedding_model.encode("test"))

            # Recreate collection if needed
            try:
                self.client.recreate_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
            except Exception:
                # Collection might already exist, that's OK
                pass

        except Exception as e:
            raise RuntimeError(f"Failed to initialize Qdrant: {e}")

    def upsert(self, documents: list[Document]) -> dict:
        """Upsert documents into Qdrant."""
        from qdrant_client.models import PointStruct

        unique_docs, duplicates = self._deduplicate_documents(documents)

        if not unique_docs:
            return {
                "status": "success",
                "upserted_count": 0,
                "duplicate_count": duplicates,
            }

        points = []
        for idx, doc in enumerate(unique_docs):
            embedding = self.embedding_model.encode(doc.page_content).tolist()
            point = PointStruct(
                id=hash(doc.metadata.get("source_file", "") + str(idx)) & 0x7FFFFFFF,
                vector=embedding,
                payload={
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                },
            )
            points.append(point)

        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            return {
                "status": "success",
                "upserted_count": len(unique_docs),
                "duplicate_count": duplicates,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to upsert to Qdrant: {e}")

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filters: Optional[dict] = None,
        mode: str = "semantic",
    ) -> list[VectorStoreResult]:
        """Search Qdrant for relevant documents with hybrid search support."""
        try:
            if mode == "hybrid":
                # Get all documents for hybrid search
                all_docs = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=10000,
                )

                if not all_docs[0]:
                    return []

                # Convert to hybrid searcher format
                documents = []
                for point in all_docs[0]:
                    documents.append({
                        "id": str(point.id),
                        "content": point.payload.get("content", ""),
                        "metadata": point.payload.get("metadata", {}),
                    })

                from rag_pipeline.hybrid_search import HybridSearcher
                try:
                    searcher = HybridSearcher(self.embedding_model, documents)
                    hybrid_results = searcher.search(query, documents, top_k, mode="hybrid")
                except Exception:
                    # Fall back to semantic search
                    return self._semantic_search_qdrant(query, top_k, min_score)

                # Convert results
                search_results = []
                for doc_id, score in hybrid_results:
                    if score >= min_score:
                        # Find the document
                        for doc in documents:
                            if doc["id"] == doc_id:
                                result = VectorStoreResult(
                                    document_id=doc_id,
                                    content=doc["content"],
                                    metadata=doc["metadata"],
                                    similarity_score=score,
                                )
                                search_results.append(result)
                                break

                return search_results
            else:
                return self._semantic_search_qdrant(query, top_k, min_score)

        except Exception as e:
            raise RuntimeError(f"Failed to search Qdrant: {e}")

    def _semantic_search_qdrant(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[VectorStoreResult]:
        """Semantic search for Qdrant."""
        query_embedding = self.embedding_model.encode(query).tolist()

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            score_threshold=min_score,
        )

        search_results = []
        for hit in results:
            result = VectorStoreResult(
                document_id=str(hit.id),
                content=hit.payload.get("content", ""),
                metadata=hit.payload.get("metadata", {}),
                similarity_score=hit.score,
            )
            search_results.append(result)

        return search_results

    def delete(self, document_ids: list[str]) -> dict:
        """Delete documents from Qdrant."""
        try:
            point_ids = [int(doc_id) for doc_id in document_ids]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=point_ids,
            )
            return {
                "status": "success",
                "deleted_count": len(document_ids),
            }
        except Exception as e:
            raise RuntimeError(f"Failed to delete from Qdrant: {e}")

    def get_stats(self) -> dict:
        """Get Qdrant collection statistics."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "document_count": collection_info.points_count,
                "vector_store_type": "qdrant",
                "url": settings.qdrant_url,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get Qdrant stats: {e}")

    def clear_all(self) -> dict:
        """Clear all documents from Qdrant collection."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            # Recreate empty collection
            from qdrant_client.models import Distance, VectorParams
            embedding_dim = len(self.embedding_model.encode("test"))
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE,
                ),
            )
            return {
                "status": "success",
                "cleared_count": 0,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to clear Qdrant: {e}")


class VectorStoreFactory:
    """Factory for creating vector store instances."""

    STORES = {
        "chromadb": ChromaDBVectorStore,
        "qdrant": QdrantVectorStore,
    }

    @classmethod
    def create(
        cls,
        embedding_model,
        store_type: Optional[str] = None,
        collection_name: str = "rag_documents",
    ) -> BaseVectorStore:
        """Create vector store instance."""
        store_type = store_type or settings.vector_store_type

        if store_type not in cls.STORES:
            raise ValueError(f"Unknown vector store type: {store_type}")

        store_class = cls.STORES[store_type]
        return store_class(embedding_model, collection_name)
