from types import SimpleNamespace

from qdrant_client import QdrantClient

from rag_pipeline.vector_store import BaseVectorStore, QdrantVectorStore, _qdrant_filter


class FakeBatch(list):
    def tolist(self):
        return list(self)


class FakeVector(list):
    def tolist(self):
        return list(self)


class FakeEmbeddingModel:
    def encode(self, value):
        if isinstance(value, list):
            return FakeBatch([[1.0, 0.0] for _ in value])
        return FakeVector([1.0, 0.0])


def test_existing_qdrant_collection_is_never_recreated(monkeypatch):
    calls = []

    class ExistingCollectionClient:
        def __init__(self, **kwargs):
            pass

        def collection_exists(self, collection_name):
            return True

        def create_collection(self, **kwargs):
            calls.append("create")

        def recreate_collection(self, **kwargs):
            calls.append("recreate")

    monkeypatch.setattr(QdrantClient, "__init__", ExistingCollectionClient.__init__)
    monkeypatch.setattr(
        QdrantClient, "collection_exists", ExistingCollectionClient.collection_exists
    )
    monkeypatch.setattr(
        QdrantClient, "create_collection", ExistingCollectionClient.create_collection
    )
    monkeypatch.setattr(
        QdrantClient, "recreate_collection", ExistingCollectionClient.recreate_collection
    )

    QdrantVectorStore(FakeEmbeddingModel(), collection_name="existing")

    assert calls == []


def test_qdrant_filter_targets_metadata_tenant_field():
    query_filter = _qdrant_filter({"tenant_id": "tenant-a"})

    assert query_filter.must[0].key == "metadata.tenant_id"
    assert query_filter.must[0].match.value == "tenant-a"


def test_qdrant_point_ids_are_stable_for_chunk_ids():
    store = object.__new__(QdrantVectorStore)
    BaseVectorStore.__init__(store, FakeEmbeddingModel())
    captured = []

    class FakeClient:
        def upsert(self, collection_name, points):
            captured.extend(point.id for point in points)

    store.client = FakeClient()
    document = SimpleNamespace(
        page_content="stable text",
        metadata={
            "tenant_id": "tenant-a",
            "document_id": "doc-1",
            "unique_chunk_id": "doc-1_0",
        },
    )

    store.upsert([document])
    first_id = captured[-1]
    store.upsert([document])

    assert captured[-1] == first_id
    assert first_id >= 0


def test_qdrant_semantic_search_uses_query_points_and_tenant_filter():
    store = object.__new__(QdrantVectorStore)
    BaseVectorStore.__init__(store, FakeEmbeddingModel())

    class FakeClient:
        def query_points(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(points=[SimpleNamespace(
                id=7,
                score=0.9,
                payload={"content": "tenant document", "metadata": {"tenant_id": "a"}},
            )])

    store.client = FakeClient()
    results = store._semantic_search_qdrant(
        "query", top_k=3, min_score=0.1, filters={"tenant_id": "a"}
    )

    assert len(results) == 1
    assert results[0].document_id == "7"
    assert results[0].metadata["tenant_id"] == "a"
    assert store.client.kwargs["query"] == [1.0, 0.0]
    assert store.client.kwargs["query_filter"].must[0].key == "metadata.tenant_id"
