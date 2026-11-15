try:
    from langchain.schema import Document
except ImportError:
    from langchain_core.documents import Document

from rag_pipeline.vector_store import BaseVectorStore, ChromaDBVectorStore


class FakeVector(list):
    def tolist(self):
        return list(self)


class FakeVectorBatch(list):
    def tolist(self):
        return [vector.tolist() for vector in self]


class FakeEmbeddingModel:
    def __init__(self):
        self.calls = []

    def encode(self, value):
        self.calls.append(value)
        if isinstance(value, list):
            return FakeVectorBatch(FakeVector([1.0, 0.0]) for _ in value)
        return FakeVector([1.0, 0.0])


class FakeChromaCollection:
    def __init__(self, records):
        self.records = list(records)
        self.get_count = 0

    def _matches(self, metadata, where):
        if not where:
            return True
        if "$and" in where:
            return all(self._matches(metadata, item) for item in where["$and"])
        return all(metadata.get(key) == value for key, value in where.items())

    def get(self, include=None, where=None):
        self.get_count += 1
        records = [r for r in self.records if self._matches(r["metadata"], where)]
        return {
            "ids": [r["id"] for r in records],
            "documents": [r["content"] for r in records],
            "metadatas": [r["metadata"] for r in records],
        }

    def query(self, query_embeddings, n_results, where=None):
        records = [r for r in self.records if self._matches(r["metadata"], where)][:n_results]
        return {
            "ids": [[r["id"] for r in records]],
            "documents": [[r["content"] for r in records]],
            "metadatas": [[r["metadata"] for r in records]],
            "distances": [[0.1 for _ in records]],
        }

    def upsert(self, ids, documents, metadatas, embeddings):
        by_id = {record["id"]: record for record in self.records}
        for doc_id, content, metadata in zip(ids, documents, metadatas):
            by_id[doc_id] = {"id": doc_id, "content": content, "metadata": metadata}
        self.records = list(by_id.values())

    def delete(self, ids):
        removed = set(ids)
        self.records = [record for record in self.records if record["id"] not in removed]

    def count(self, where=None):
        return sum(self._matches(r["metadata"], where) for r in self.records)


def make_store():
    model = FakeEmbeddingModel()
    store = object.__new__(ChromaDBVectorStore)
    BaseVectorStore.__init__(store, model)
    store.collection = FakeChromaCollection([
        {"id": "red-1", "content": "quasar astronomy telescope", "metadata": {
            "tenant_id": "red", "document_id": "doc-red-1", "unique_chunk_id": "red-1"
        }},
        {"id": "red-2", "content": "pasta cooking recipe", "metadata": {
            "tenant_id": "red", "document_id": "doc-red-2", "unique_chunk_id": "red-2"
        }},
        {"id": "red-3", "content": "garden flowers soil", "metadata": {
            "tenant_id": "red", "document_id": "doc-red-3", "unique_chunk_id": "red-3"
        }},
        {"id": "blue-1", "content": "quasar physics stellar object", "metadata": {
            "tenant_id": "blue", "document_id": "doc-blue-1", "unique_chunk_id": "blue-1"
        }},
    ])
    return store


def test_keyword_search_uses_bm25_and_respects_tenant_filter():
    store = make_store()

    results = store.search("quasar", top_k=1, mode="keyword", filters={"tenant_id": "red"})

    assert len(results) == 1
    assert results[0].metadata["document_id"] == "doc-red-1"
    assert results[0].metadata["tenant_id"] == "red"


def test_hybrid_search_fuses_rankings_without_reembedding_the_corpus():
    store = make_store()

    results = store.search("quasar", top_k=2, mode="hybrid", filters={"tenant_id": "red"})

    assert results
    assert all(result.metadata["tenant_id"] == "red" for result in results)
    assert store.embedding_model.calls == ["quasar"]
    assert store.collection.get_count == 1


def test_bm25_cache_invalidates_after_upsert_and_document_delete():
    store = make_store()
    store.search("nebula", top_k=5, mode="keyword")
    first_get_count = store.collection.get_count

    document = Document(
        page_content="bright nebula astronomy",
        metadata={
            "tenant_id": "red",
            "document_id": "doc-new",
            "unique_chunk_id": "new-1",
            "chunk_index": 0,
        },
    )
    store.upsert([document])
    results = store.search("nebula", top_k=1, mode="keyword", filters={"tenant_id": "red"})

    assert store.collection.get_count == first_get_count + 1
    assert results[0].metadata["document_id"] == "doc-new"
    deleted = store.delete_by_document_id("doc-new", tenant_id="red")
    assert deleted["deleted_count"] == 1
