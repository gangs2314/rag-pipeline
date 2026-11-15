"""Vector store abstraction layer supporting ChromaDB and Qdrant."""

import hashlib
import json
from abc import ABC, abstractmethod
from threading import RLock
from typing import Optional

try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

from rag_pipeline.config import settings
from rag_pipeline.deduplication import deduplicate_documents
from rag_pipeline.retrieval.bm25_index import BM25Index
from rag_pipeline.retrieval.fusion import rrf_fuse


def _combine_chroma_filter(
    filters: Optional[dict], required: Optional[dict] = None
) -> Optional[dict]:
    conditions = []
    if filters:
        conditions.append(filters)
    if required:
        conditions.extend({key: value} for key, value in required.items())
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def _matches_filters(metadata: dict, filters: Optional[dict]) -> bool:
    """Evaluate the equality and boolean filters used by the portable API layer."""
    if not filters:
        return True
    if "$and" in filters:
        return all(_matches_filters(metadata, item) for item in filters["$and"])
    if "$or" in filters:
        return any(_matches_filters(metadata, item) for item in filters["$or"])
    for key, expected in filters.items():
        actual = metadata.get(key)
        if isinstance(expected, dict):
            if "$eq" in expected and actual != expected["$eq"]:
                return False
            if "$ne" in expected and actual == expected["$ne"]:
                return False
            if "$in" in expected and actual not in expected["$in"]:
                return False
        elif actual != expected:
            return False
    return True


def _qdrant_filter(filters: Optional[dict]):
    if not filters:
        return None
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    must = []
    for key, value in filters.items():
        if key in {"$and", "$or"}:
            # Tenant filters are scalar equality filters; reject unsupported
            # expressions rather than accidentally widening a tenant query.
            raise ValueError("Qdrant supports scalar equality filters in this API")
        must.append(FieldCondition(key=f"metadata.{key}", match=MatchValue(value=value)))
    return Filter(must=must)


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
        self._documents_cache: Optional[list[dict]] = None
        self._bm25_indexes: dict[str, BM25Index] = {}
        self._cache_lock = RLock()

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
    def get_stats(self, filters: Optional[dict] = None) -> dict:
        """Get vector store statistics."""
        pass

    @abstractmethod
    def clear_all(self) -> dict:
        """Clear all documents from vector store."""
        pass

    @abstractmethod
    def delete_by_document_id(
        self, document_id: str, tenant_id: Optional[str] = None
    ) -> dict:
        """Delete all chunks belonging to one document, optionally tenant-scoped."""
        pass

    def delete_by_filters(self, filters: dict) -> dict:
        """Delete all chunks matching metadata filters."""
        raise NotImplementedError

    def _load_documents(self) -> list[dict]:
        raise NotImplementedError

    def _all_documents(self) -> list[dict]:
        with self._cache_lock:
            if self._documents_cache is None:
                self._documents_cache = self._load_documents()
            return self._documents_cache

    def _invalidate_search_cache(self) -> None:
        with self._cache_lock:
            self._documents_cache = None
            self._bm25_indexes.clear()

    def _bm25_index(self, filters: Optional[dict]) -> BM25Index:
        cache_key = json.dumps(filters or {}, sort_keys=True, default=str)
        with self._cache_lock:
            if cache_key not in self._bm25_indexes:
                documents = [
                    document
                    for document in self._all_documents()
                    if _matches_filters(document.get("metadata") or {}, filters)
                ]
                index = BM25Index()
                index.build(documents)
                self._bm25_indexes[cache_key] = index
            return self._bm25_indexes[cache_key]

    def _keyword_search(
        self, query: str, top_k: int, filters: Optional[dict]
    ) -> list[VectorStoreResult]:
        results = self._bm25_index(filters).search(query, top_k=top_k)
        if not results:
            return []
        max_score = max(result.get("bm25_score", 0.0) for result in results)
        return [
            VectorStoreResult(
                document_id=str(result.get("id", result.get("document_id", ""))),
                content=result.get("content", ""),
                metadata=result.get("metadata") or {},
                similarity_score=(
                    float(result.get("bm25_score", 0.0)) / max_score if max_score > 0 else 0.0
                ),
            )
            for result in results
        ]

    def _dense_search(
        self, query: str, top_k: int, min_score: float, filters: Optional[dict]
    ) -> list[VectorStoreResult]:
        return self.search(query, top_k, min_score, filters, mode="semantic")

    def _hybrid_search(
        self,
        query: str,
        top_k: int,
        min_score: float,
        filters: Optional[dict],
    ) -> list[VectorStoreResult]:
        candidate_k = max(top_k * 4, 20)
        dense = self._dense_search(query, candidate_k, min_score, filters)
        sparse = self._keyword_search(query, candidate_k, filters)
        fused = rrf_fuse(
            [[result.to_dict() for result in dense], [result.to_dict() for result in sparse]],
            k=60,
        )
        return [
            VectorStoreResult(
                document_id=str(result["document_id"]),
                content=result.get("content", ""),
                metadata=result.get("metadata") or {},
                similarity_score=float(result.get("rrf_score", 0.0)),
            )
            for result in fused[:top_k]
        ]

    def _deduplicate_documents(
        self,
        documents: list[Document],
    ) -> tuple[list[Document], int]:
        """Remove exact duplicate chunks within the same source scope using sets."""
        return deduplicate_documents(documents)


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
        """Idempotently upsert chunks, using set-based exact deduplication."""
        unique_docs, duplicate_count = self._deduplicate_documents(documents)
        if not unique_docs:
            return {
                "status": "success",
                "upserted_count": 0,
                "duplicate_count": duplicate_count,
            }

        ids = [
            str(
                doc.metadata.get("unique_chunk_id")
                or self._stable_chunk_id(doc.page_content, doc.metadata)
            )
            for doc in unique_docs
        ]
        embeddings = self.embedding_model.encode([doc.page_content for doc in unique_docs]).tolist()
        try:
            self.collection.upsert(
                ids=ids,
                documents=[doc.page_content for doc in unique_docs],
                metadatas=[doc.metadata for doc in unique_docs],
                embeddings=embeddings,
            )
            self._invalidate_search_cache()
            return {
                "status": "success", "upserted_count": len(ids), "duplicate_count": duplicate_count
            }
        except Exception as e:
            raise RuntimeError(f"Failed to upsert documents to ChromaDB: {e}")

    @staticmethod
    def _stable_chunk_id(content: str, metadata: dict) -> str:
        key = "|".join(
            str(value)
            for value in (
                metadata.get("tenant_id", "public"),
                metadata.get("document_id", metadata.get("source_file", "unknown")),
                metadata.get("chunk_index", ""),
                hashlib.sha256(content.encode("utf-8")).hexdigest(),
            )
        )
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _load_documents(self) -> list[dict]:
        data = self.collection.get(include=["documents", "metadatas"])
        ids = data.get("ids") or []
        contents = data.get("documents") or []
        metadatas = data.get("metadatas") or []
        return [
            {
                "id": str(doc_id),
                "content": contents[index] if index < len(contents) else "",
                "metadata": metadatas[index] if index < len(metadatas) else {},
            }
            for index, doc_id in enumerate(ids)
        ]

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
            if top_k <= 0:
                return []
            if mode == "hybrid":
                return self._hybrid_search(query, top_k, min_score, filters)
            if mode == "keyword":
                return self._keyword_search(query, top_k, filters)
            if mode == "semantic":
                return self._semantic_search(query, top_k, min_score, filters)
            raise ValueError(f"Unknown search mode: {mode}")
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
            where=_combine_chroma_filter(filters),
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
        return super()._hybrid_search(query, top_k, min_score, filters)

    def delete(self, document_ids: list[str]) -> dict:
        """Delete documents from ChromaDB."""
        try:
            self.collection.delete(ids=document_ids)
            self._invalidate_search_cache()
            return {
                "status": "success",
                "deleted_count": len(document_ids),
            }
        except Exception as e:
            raise RuntimeError(f"Failed to delete from ChromaDB: {e}")

    def delete_by_document_id(
        self, document_id: str, tenant_id: Optional[str] = None
    ) -> dict:
        required = {"document_id": document_id}
        if tenant_id is not None:
            required["tenant_id"] = tenant_id
        return self.delete_by_filters(required)

    def delete_by_filters(self, filters: dict) -> dict:
        try:
            where = _combine_chroma_filter(filters)
            matches = self.collection.get(where=where, include=["metadatas"])
            ids = matches.get("ids") or []
            if ids:
                self.collection.delete(ids=ids)
            self._invalidate_search_cache()
            return {"status": "success", "deleted_count": len(ids)}
        except Exception as e:
            raise RuntimeError(f"Failed to delete matching ChromaDB documents: {e}")

    def get_stats(self, filters: Optional[dict] = None) -> dict:
        """Get ChromaDB collection statistics."""
        try:
            where = _combine_chroma_filter(filters)
            count = self.collection.count(where=where) if where else self.collection.count()
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

            self._invalidate_search_cache()

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

            # Never recreate an existing collection here: recreation deletes all
            # indexed data. Create it only when this is a first-time deployment.
            if not self.client.collection_exists(collection_name=collection_name):
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )

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
        embeddings = self.embedding_model.encode([doc.page_content for doc in unique_docs]).tolist()
        for doc, embedding in zip(unique_docs, embeddings):
            chunk_key = str(
                doc.metadata.get("unique_chunk_id")
                or ChromaDBVectorStore._stable_chunk_id(doc.page_content, doc.metadata)
            )
            point_id = int.from_bytes(hashlib.sha256(chunk_key.encode("utf-8")).digest()[:8], "big")
            point_id &= (1 << 63) - 1
            point = PointStruct(
                id=point_id,
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
            self._invalidate_search_cache()
            return {
                "status": "success",
                "upserted_count": len(unique_docs),
                "duplicate_count": duplicates,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to upsert to Qdrant: {e}")

    def _scroll_points(self, filters: Optional[dict] = None) -> list:
        points = []
        offset = None
        while True:
            page, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
                scroll_filter=_qdrant_filter(filters),
            )
            points.extend(page)
            if offset is None:
                return points

    def _load_documents(self) -> list[dict]:
        return [
            {
                "id": str(point.id),
                "content": (point.payload or {}).get("content", ""),
                "metadata": (point.payload or {}).get("metadata", {}),
            }
            for point in self._scroll_points()
        ]

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
            if top_k <= 0:
                return []
            if mode == "hybrid":
                return self._hybrid_search(query, top_k, min_score, filters)
            if mode == "keyword":
                return self._keyword_search(query, top_k, filters)
            if mode == "semantic":
                return self._semantic_search_qdrant(query, top_k, min_score, filters)
            raise ValueError(f"Unknown search mode: {mode}")

        except Exception as e:
            raise RuntimeError(f"Failed to search Qdrant: {e}")

    def _semantic_search_qdrant(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        filters: Optional[dict] = None,
    ) -> list[VectorStoreResult]:
        """Semantic search for Qdrant."""
        query_embedding = self.embedding_model.encode(query).tolist()

        query_filter = _qdrant_filter(filters)
        if hasattr(self.client, "query_points"):
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_embedding,
                limit=top_k,
                score_threshold=min_score,
                query_filter=query_filter,
                with_payload=True,
                with_vectors=False,
            ).points
        else:
            # Compatibility with qdrant-client versions that still expose search().
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=top_k,
                score_threshold=min_score,
                query_filter=query_filter,
            )

        search_results = []
        for hit in results:
            result = VectorStoreResult(
                document_id=str(hit.id),
                content=(hit.payload or {}).get("content", ""),
                metadata=(hit.payload or {}).get("metadata", {}),
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
            self._invalidate_search_cache()
            return {
                "status": "success",
                "deleted_count": len(document_ids),
            }
        except Exception as e:
            raise RuntimeError(f"Failed to delete from Qdrant: {e}")

    def delete_by_document_id(
        self, document_id: str, tenant_id: Optional[str] = None
    ) -> dict:
        filters = {"document_id": document_id}
        if tenant_id is not None:
            filters["tenant_id"] = tenant_id
        return self.delete_by_filters(filters)

    def delete_by_filters(self, filters: dict) -> dict:
        try:
            points = self._scroll_points(filters)
            point_ids = [point.id for point in points]
            if point_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=point_ids,
                )
            self._invalidate_search_cache()
            return {"status": "success", "deleted_count": len(point_ids)}
        except Exception as e:
            raise RuntimeError(f"Failed to delete matching Qdrant documents: {e}")

    def get_stats(self, filters: Optional[dict] = None) -> dict:
        """Get Qdrant collection statistics."""
        try:
            count = self.client.count(
                collection_name=self.collection_name,
                count_filter=_qdrant_filter(filters),
                exact=True,
            ).count
            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "vector_store_type": "qdrant",
                "url": settings.qdrant_url,
            }
        except Exception as e:
            raise RuntimeError(f"Failed to get Qdrant stats: {e}")

    def clear_all(self) -> dict:
        """Clear all documents from Qdrant collection."""
        return self.delete_by_filters({})


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
