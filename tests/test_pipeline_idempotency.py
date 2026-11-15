from rag_pipeline.config import settings
from rag_pipeline.pipeline import RAGPipeline


class IdempotentFakeStore:
    def __init__(self):
        self.chunks = {}

    def upsert(self, chunks):
        for chunk in chunks:
            self.chunks[chunk.metadata["unique_chunk_id"]] = chunk
        return {"upserted_count": len(chunks), "duplicate_count": 0}


def test_reingesting_same_file_is_stable_and_tenant_scoped(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "use_parent_child_chunking", False)
    file_path = tmp_path / "notes.txt"
    file_path.write_text("A short document about stable ingestion IDs.", encoding="utf-8")

    pipeline = object.__new__(RAGPipeline)
    pipeline.vector_store = IdempotentFakeStore()
    pipeline.document_registry = {}

    first = pipeline.ingest_document(file_path, tenant_id="tenant-a", source_name="notes.txt")
    first_chunk_ids = set(pipeline.vector_store.chunks)
    second = pipeline.ingest_document(file_path, tenant_id="tenant-a", source_name="notes.txt")

    assert first["document_id"] == second["document_id"]
    assert set(pipeline.vector_store.chunks) == first_chunk_ids
    assert all(
        chunk.metadata["tenant_id"] == "tenant-a"
        for chunk in pipeline.vector_store.chunks.values()
    )

    other_tenant = pipeline.ingest_document(
        file_path, tenant_id="tenant-b", source_name="notes.txt"
    )
    assert other_tenant["document_id"] != first["document_id"]


def test_query_uses_tenant_filter_and_expanded_mmr_candidates(monkeypatch):
    monkeypatch.setattr(settings, "api_keys", "key:tenant-a")
    monkeypatch.setattr(settings, "rerank_candidate_k", 30)

    class FakeResult:
        def to_dict(self):
            return {"document_id": "chunk-1", "content": "relevant", "metadata": {}}

    class FakeStore:
        def search(self, **kwargs):
            self.kwargs = kwargs
            return [FakeResult()]

    class FakeReranker:
        def rerank(self, query, candidates, top_k):
            self.top_k = top_k
            return candidates[:top_k]

    pipeline = object.__new__(RAGPipeline)
    pipeline.vector_store = FakeStore()
    pipeline.embedding_model = object()
    reranker = FakeReranker()
    monkeypatch.setattr("rag_pipeline.pipeline.get_reranker", lambda *args, **kwargs: reranker)

    result = pipeline.query(
        "find evidence", top_k=3, mode="hybrid", rerank=True, tenant_id="tenant-a"
    )

    assert pipeline.vector_store.kwargs["top_k"] == 30
    assert pipeline.vector_store.kwargs["filters"] == {"tenant_id": "tenant-a"}
    assert pipeline.vector_store.kwargs["mode"] == "hybrid"
    assert reranker.top_k == 3
    assert result["reranked"] is True
