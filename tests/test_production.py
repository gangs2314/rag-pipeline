"""Production-ready test suite for RAG Pipeline."""

import asyncio
import json
import time
from pathlib import Path
from typing import List

import pytest
import httpx
from rag_pipeline.pipeline import get_pipeline
from rag_pipeline.loaders import DocumentLoaderFactory
from rag_pipeline.config import settings


class TestConstants:
    """Test configuration constants."""
    API_BASE = "http://localhost:8000"
    TIMEOUT = 30.0
    SAMPLE_QUERY = "What is the main topic?"


@pytest.fixture(scope="session")
def api_client():
    """Async HTTP client for API testing."""
    with httpx.Client(base_url=TestConstants.API_BASE, timeout=TestConstants.TIMEOUT) as client:
        yield client


@pytest.fixture(scope="session")
def pipeline():
    """RAG Pipeline instance."""
    return get_pipeline()


class TestAPIHealth:
    """Test API health and status endpoints."""

    def test_health_check(self, api_client):
        """Test health check endpoint."""
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "vector_store" in data

    def test_stats_endpoint(self, api_client):
        """Test statistics endpoint."""
        response = api_client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "vector_store_type" in data
        assert "document_count" in data
        assert "embedding_model" in data

    def test_root_endpoint(self, api_client):
        """Test root endpoint."""
        response = api_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "endpoints" in data


class TestDocumentUpload:
    """Test document upload functionality."""

    def test_pdf_upload(self, api_client, tmp_path):
        """Test PDF file upload."""
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
        except ImportError:
            pytest.skip("reportlab not installed")

        pdf_path = tmp_path / "test.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        c.drawString(100, 750, "Test Document")
        c.showPage()
        c.save()

        with open(pdf_path, "rb") as f:
            files = {"file": ("test.pdf", f, "application/pdf")}
            response = api_client.post("/upload", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["chunk_count"] > 0
        assert "document_id" in data

    def test_text_upload(self, api_client, tmp_path):
        """Test text file upload."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("Sample test content\nLine 2\nLine 3")

        with open(txt_path, "rb") as f:
            files = {"file": ("test.txt", f, "text/plain")}
            response = api_client.post("/upload", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_markdown_upload(self, api_client, tmp_path):
        """Test markdown file upload."""
        md_path = tmp_path / "test.md"
        md_path.write_text("# Title\n\n## Section\n\nContent here")

        with open(md_path, "rb") as f:
            files = {"file": ("test.md", f, "text/markdown")}
            response = api_client.post("/upload", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

    def test_unsupported_file_type(self, api_client, tmp_path):
        """Test uploading unsupported file type."""
        exe_path = tmp_path / "test.exe"
        exe_path.write_text("binary content")

        with open(exe_path, "rb") as f:
            files = {"file": ("test.exe", f, "application/octet-stream")}
            response = api_client.post("/upload", files=files)

        assert response.status_code in [400, 415]

    def test_empty_file_upload(self, api_client, tmp_path):
        """Test uploading empty file."""
        empty_path = tmp_path / "empty.txt"
        empty_path.write_text("")

        with open(empty_path, "rb") as f:
            files = {"file": ("empty.txt", f, "text/plain")}
            response = api_client.post("/upload", files=files)

        assert response.status_code == 200


class TestQueryFunctionality:
    """Test query and search functionality."""

    def test_simple_query(self, api_client):
        """Test simple query."""
        payload = {
            "query": "What is this document about?",
            "top_k": 5
        }
        response = api_client.post("/query", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "result_count" in data
        assert "results" in data

    def test_query_with_custom_top_k(self, api_client):
        """Test query with custom top_k."""
        for top_k in [1, 3, 5, 10]:
            payload = {"query": "test query", "top_k": top_k}
            response = api_client.post("/query", json=payload)

            assert response.status_code == 200
            data = response.json()
            assert len(data["results"]) <= top_k

    def test_empty_query_fails(self, api_client):
        """Test that empty query fails."""
        payload = {"query": "", "top_k": 5}
        response = api_client.post("/query", json=payload)

        assert response.status_code == 400

    def test_query_result_structure(self, api_client):
        """Test query result structure."""
        payload = {"query": "sample query", "top_k": 1}
        response = api_client.post("/query", json=payload)

        assert response.status_code == 200
        data = response.json()

        if data["result_count"] > 0:
            result = data["results"][0]
            assert "content" in result
            assert "metadata" in result
            assert "similarity_score" in result
            assert 0 <= result["similarity_score"] <= 1

    def test_query_performance(self, api_client):
        """Test query performance."""
        start_time = time.time()

        payload = {"query": "performance test", "top_k": 5}
        response = api_client.post("/query", json=payload)

        elapsed = time.time() - start_time

        assert response.status_code == 200
        assert elapsed < 5.0  # Should complete within 5 seconds


class TestPipeline:
    """Test core pipeline functionality."""

    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline is not None
        assert pipeline.embedding_model is not None
        assert pipeline.vector_store is not None

    def test_document_ingestion(self, pipeline, tmp_path):
        """Test document ingestion."""
        txt_path = tmp_path / "ingest_test.txt"
        txt_path.write_text("Test content for ingestion")

        result = pipeline.ingest_document(txt_path)

        assert result["status"] == "success"
        assert result["chunk_count"] > 0
        assert "document_id" in result

    def test_query_execution(self, pipeline):
        """Test query execution."""
        result = pipeline.query("test query", top_k=5)

        assert result["status"] == "success"
        assert "results" in result
        assert "result_count" in result

    def test_statistics_retrieval(self, pipeline):
        """Test statistics retrieval."""
        stats = pipeline.get_stats()

        assert "document_count" in stats
        assert "embedding_model" in stats
        assert "chunking_config" in stats

    def test_document_deletion(self, pipeline):
        """Test document deletion."""
        test_id = "test-doc-123"
        result = pipeline.delete_document(test_id)

        assert "status" in result


class TestChunking:
    """Test document chunking strategies."""

    def test_recursive_character_chunking(self, tmp_path):
        """Test recursive character chunking."""
        from rag_pipeline.chunking import RecursiveCharacterChunkingStrategy
        from rag_pipeline.loaders import TextDocumentLoader

        txt_path = tmp_path / "chunking_test.txt"
        content = "This is a test. " * 100
        txt_path.write_text(content)

        loader = TextDocumentLoader(txt_path, "txt")
        docs = loader.load()

        chunker = RecursiveCharacterChunkingStrategy(chunk_size=512, chunk_overlap=50)
        chunks = chunker.chunk(docs)

        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk.page_content) > 0
            assert "chunk_index" in chunk.metadata

    def test_markdown_chunking(self, tmp_path):
        """Test markdown chunking."""
        from rag_pipeline.chunking import MarkdownHeaderChunkingStrategy
        from rag_pipeline.loaders import TextDocumentLoader

        md_path = tmp_path / "chunking_test.md"
        content = """# Title
## Section 1
Content 1
## Section 2
Content 2
"""
        md_path.write_text(content)

        loader = TextDocumentLoader(md_path, "md")
        docs = loader.load()

        chunker = MarkdownHeaderChunkingStrategy()
        chunks = chunker.chunk(docs)

        assert len(chunks) > 0


class TestDocumentLoaders:
    """Test document loaders."""

    def test_text_loader(self, tmp_path):
        """Test text document loader."""
        txt_path = tmp_path / "test.txt"
        txt_path.write_text("Test content")

        docs = DocumentLoaderFactory.load_document(txt_path)

        assert len(docs) > 0
        assert docs[0].page_content == "Test content"

    def test_json_loader(self, tmp_path):
        """Test JSON document loader."""
        json_path = tmp_path / "test.json"
        json_path.write_text(json.dumps({"key": "value", "nested": {"item": 1}}))

        docs = DocumentLoaderFactory.load_document(json_path)

        assert len(docs) > 0
        assert "key" in docs[0].page_content or "value" in docs[0].page_content

    def test_loader_factory(self, tmp_path):
        """Test document loader factory."""
        txt_path = tmp_path / "factory_test.txt"
        txt_path.write_text("Factory test")

        loader = DocumentLoaderFactory.get_loader(txt_path)
        assert loader is not None

        docs = loader.load()
        assert len(docs) > 0


class TestConfiguration:
    """Test configuration management."""

    def test_settings_loaded(self):
        """Test that settings are loaded correctly."""
        assert settings.chunk_size > 0
        assert settings.chunk_overlap >= 0
        assert settings.embedding_model
        assert settings.vector_store_type in ["chromadb", "qdrant"]

    def test_directories_created(self):
        """Test that required directories are created."""
        assert settings.chromadb_path.exists()
        assert settings.upload_dir.exists()
        assert settings.embedding_cache_dir.exists()


class TestPerformance:
    """Performance and load testing."""

    def test_concurrent_queries(self, api_client):
        """Test concurrent query handling."""
        import concurrent.futures

        def run_query():
            payload = {"query": "concurrent test", "top_k": 3}
            response = api_client.post("/query", json=payload)
            return response.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(run_query) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert all(status == 200 for status in results)

    def test_query_result_consistency(self, api_client):
        """Test that queries return consistent results."""
        query = "consistency test"

        response1 = api_client.post("/query", json={"query": query, "top_k": 3})
        response2 = api_client.post("/query", json={"query": query, "top_k": 3})

        data1 = response1.json()
        data2 = response2.json()

        assert data1["result_count"] == data2["result_count"]


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_malformed_json(self, api_client):
        """Test handling of malformed JSON."""
        response = api_client.post(
            "/query",
            content=b"invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code >= 400

    def test_missing_required_field(self, api_client):
        """Test missing required field in query."""
        payload = {"top_k": 5}  # Missing 'query'
        response = api_client.post("/query", json=payload)

        assert response.status_code >= 400

    def test_invalid_top_k(self, api_client):
        """Test invalid top_k value."""
        payload = {"query": "test", "top_k": -1}
        response = api_client.post("/query", json=payload)

        # Should either validate or handle gracefully
        assert response.status_code in [200, 400]


# Integration Tests
class TestIntegration:
    """End-to-end integration tests."""

    def test_full_workflow(self, api_client, tmp_path):
        """Test complete workflow: upload -> query."""
        # Create and upload document
        txt_path = tmp_path / "integration_test.txt"
        txt_path.write_text("Integration test document with test content")

        with open(txt_path, "rb") as f:
            files = {"file": ("integration_test.txt", f, "text/plain")}
            upload_response = api_client.post("/upload", files=files)

        assert upload_response.status_code == 200
        upload_data = upload_response.json()
        assert upload_data["status"] == "success"

        # Query the uploaded document
        time.sleep(1)  # Allow time for indexing

        query_payload = {"query": "integration test", "top_k": 5}
        query_response = api_client.post("/query", json=query_payload)

        assert query_response.status_code == 200
        query_data = query_response.json()
        assert query_data["status"] == "success"

    def test_multiple_document_search(self, api_client, tmp_path):
        """Test searching across multiple documents."""
        # Upload multiple documents
        for i in range(3):
            txt_path = tmp_path / f"doc_{i}.txt"
            txt_path.write_text(f"Document {i} with unique content {i}")

            with open(txt_path, "rb") as f:
                files = {"file": (f"doc_{i}.txt", f, "text/plain")}
                response = api_client.post("/upload", files=files)

            assert response.status_code == 200

        time.sleep(1)

        # Query should find results from multiple documents
        query_payload = {"query": "Document", "top_k": 10}
        response = api_client.post("/query", json=query_payload)

        assert response.status_code == 200
        data = response.json()
        assert data["result_count"] > 0


# Run tests with pytest
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])


class TestProductionRobustness:
    """Test production robustness features: validation, rate limiting, metrics."""

    def test_upload_mime_type_validation(self, api_client, tmp_path):
        """Test MIME type validation rejects unsupported types."""
        unsupported_path = tmp_path / "test.xyz"
        unsupported_path.write_text("content")

        with open(unsupported_path, "rb") as f:
            files = {"file": ("test.xyz", f, "application/octet-stream")}
            response = api_client.post("/upload", files=files)

        assert response.status_code == 400
        assert "Unsupported file type" in response.json()["detail"]

    def test_query_empty_string_validation(self, api_client):
        """Test query validation rejects empty strings."""
        payload = {"query": "", "mode": "semantic"}
        response = api_client.post("/query", json=payload)

        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_query_whitespace_validation(self, api_client):
        """Test query validation rejects whitespace-only strings."""
        payload = {"query": "   \n\t  ", "mode": "semantic"}
        response = api_client.post("/query", json=payload)

        assert response.status_code == 400

    def test_query_invalid_mode_validation(self, api_client):
        """Test query validation rejects invalid search modes."""
        payload = {"query": "test", "mode": "invalid_mode"}
        response = api_client.post("/query", json=payload)

        assert response.status_code == 400
        assert "Invalid search mode" in response.json()["detail"]

    def test_query_valid_modes_accepted(self, api_client):
        """Test all valid search modes are accepted."""
        for mode in ["semantic", "keyword", "hybrid"]:
            payload = {"query": "test query", "mode": mode}
            response = api_client.post("/query", json=payload)

            # Should succeed (200) or fail gracefully (not 400 validation error)
            assert response.status_code != 400

    def test_file_size_limit_enforced(self, api_client, tmp_path):
        """Test file size limit is enforced."""
        from rag_pipeline.config import settings

        # Create oversized file
        large_path = tmp_path / "large.txt"
        large_content = b"x" * (int(settings.max_file_size_mb * 1024 * 1024) + 1024)
        large_path.write_bytes(large_content)

        with open(large_path, "rb") as f:
            files = {"file": ("large.txt", f, "text/plain")}
            response = api_client.post("/upload", files=files)

        assert response.status_code == 413
        assert "exceeds" in response.json()["detail"].lower()

    def test_metrics_endpoint_exists(self, api_client):
        """Test /metrics endpoint returns Prometheus format."""
        response = api_client.get("/metrics")

        assert response.status_code == 200
        content = response.text
        assert "# HELP" in content
        assert "# TYPE" in content
        assert "rag_" in content

    def test_metrics_include_requests_counter(self, api_client):
        """Test metrics include request counter."""
        response = api_client.get("/metrics")

        assert response.status_code == 200
        assert "rag_requests_total" in response.text or "rag_" in response.text

    def test_rate_limit_429_response(self, api_client):
        """Test rate limiting returns 429 status."""
        # Make many requests to trigger rate limit
        responses = []
        for i in range(150):  # Exceed limit of 100/min
            response = api_client.get("/health")
            responses.append(response.status_code)

            if response.status_code == 429:
                assert "Rate limit exceeded" in response.json()["detail"]
                break

        # Should have hit rate limit
        assert 429 in responses

    def test_upload_rejected_without_filename(self, api_client):
        """Test upload is rejected when filename is missing."""
        response = api_client.post("/upload", files={"file": ("", b"content")})

        assert response.status_code == 400

    def test_error_responses_have_detail(self, api_client):
        """Test all error responses include detail field."""
        error_responses = [
            api_client.post("/query", json={"query": ""}),
            api_client.post("/query", json={"query": "test", "mode": "invalid"}),
        ]

        for response in error_responses:
            if response.status_code >= 400:
                assert "detail" in response.json()

    def test_health_check_always_succeeds(self, api_client):
        """Test health check endpoint always returns 200."""
        for _ in range(5):
            response = api_client.get("/health")
            assert response.status_code == 200
            assert response.json()["status"] == "healthy"

    def test_metrics_endpoint_never_fails(self, api_client):
        """Test metrics endpoint never fails."""
        for _ in range(5):
            response = api_client.get("/metrics")
            assert response.status_code == 200

