from fastapi.testclient import TestClient

from rag_pipeline import api
from rag_pipeline.config import settings


def test_upload_filename_cannot_escape_upload_directory(monkeypatch, tmp_path):
    outside_file = tmp_path / "victim.txt"
    outside_file.write_text("keep this file", encoding="utf-8")
    upload_dir = tmp_path / "uploads"
    monkeypatch.setattr(settings, "upload_dir", upload_dir)
    monkeypatch.setattr(settings, "api_keys", None)
    monkeypatch.setattr(api, "rate_limiter", api.RateLimiter(requests_per_minute=100))

    class FakePipeline:
        def ingest_document(self, file_path, **kwargs):
            assert file_path.parent == upload_dir
            assert file_path.read_text(encoding="utf-8") == "uploaded text"
            assert kwargs["source_name"] == "victim.txt"
            return {
                "document_id": "doc-1",
                "file_name": "victim.txt",
                "chunk_count": 1,
                "status": "success",
            }

    monkeypatch.setattr(api, "get_pipeline", lambda: FakePipeline())
    client = TestClient(api.app)

    response = client.post(
        "/upload",
        files={"file": ("../victim.txt", b"uploaded text", "text/plain")},
    )

    assert response.status_code == 200
    assert outside_file.read_text(encoding="utf-8") == "keep this file"
    assert list(upload_dir.iterdir()) == []
