"""FastAPI REST API for RAG pipeline."""

import mimetypes
import time
from pathlib import Path
from typing import Optional
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline.config import settings
from rag_pipeline.pipeline import get_pipeline

# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Query request model."""

    query: str
    top_k: Optional[int] = None
    mode: Optional[str] = "semantic"  # 'semantic', 'keyword', or 'hybrid'


class QueryResponse(BaseModel):
    """Query response model."""

    query: str
    result_count: int
    results: list[dict]
    status: str


class DocumentIngestionResponse(BaseModel):
    """Document ingestion response model."""

    document_id: str
    file_name: str
    chunk_count: int
    status: str
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    vector_store: str


# Prometheus metrics
class PrometheusMetrics:
    """Simple Prometheus metrics collector."""

    def __init__(self):
        self.request_count = defaultdict(int)
        self.request_duration = defaultdict(list)
        self.error_count = defaultdict(int)
        self.upload_bytes = 0
        self.query_count = 0

    def record_request(self, endpoint: str, duration: float, status: int):
        """Record request metrics."""
        self.request_count[endpoint] += 1
        self.request_duration[endpoint].append(duration)
        if status >= 400:
            self.error_count[endpoint] += 1

    def record_upload(self, bytes_uploaded: int):
        """Record upload bytes."""
        self.upload_bytes += bytes_uploaded

    def record_query(self):
        """Record query."""
        self.query_count += 1

    def get_metrics(self) -> str:
        """Get Prometheus-formatted metrics."""
        lines = []
        lines.append("# HELP rag_requests_total Total requests by endpoint")
        lines.append("# TYPE rag_requests_total counter")
        for endpoint, count in self.request_count.items():
            lines.append(f'rag_requests_total{{endpoint="{endpoint}"}} {count}')

        lines.append("# HELP rag_errors_total Total errors by endpoint")
        lines.append("# TYPE rag_errors_total counter")
        for endpoint, count in self.error_count.items():
            lines.append(f'rag_errors_total{{endpoint="{endpoint}"}} {count}')

        lines.append("# HELP rag_request_duration_seconds Request duration")
        lines.append("# TYPE rag_request_duration_seconds gauge")
        for endpoint, durations in self.request_duration.items():
            if durations:
                avg = sum(durations) / len(durations)
                lines.append(f'rag_request_duration_seconds{{endpoint="{endpoint}"}} {avg:.4f}')

        lines.append("# HELP rag_uploads_bytes_total Total bytes uploaded")
        lines.append("# TYPE rag_uploads_bytes_total counter")
        lines.append(f"rag_uploads_bytes_total {self.upload_bytes}")

        lines.append("# HELP rag_queries_total Total queries")
        lines.append("# TYPE rag_queries_total counter")
        lines.append(f"rag_queries_total {self.query_count}")

        return "\n".join(lines)


# Rate limiter
class RateLimiter:
    """Simple in-memory rate limiter."""

    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.client_requests = defaultdict(list)

    def is_allowed(self, client_id: str) -> bool:
        """Check if client is allowed to make request."""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)

        # Clean old requests
        self.client_requests[client_id] = [
            ts for ts in self.client_requests[client_id] if ts > minute_ago
        ]

        if len(self.client_requests[client_id]) < self.requests_per_minute:
            self.client_requests[client_id].append(now)
            return True

        return False

    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client."""
        now = datetime.utcnow()
        minute_ago = now - timedelta(minutes=1)

        self.client_requests[client_id] = [
            ts for ts in self.client_requests[client_id] if ts > minute_ago
        ]

        return max(0, self.requests_per_minute - len(self.client_requests[client_id]))


# Allowed MIME types for uploads
ALLOWED_MIME_TYPES = {
    "text/plain",
    "text/markdown",
    "application/pdf",
    "application/json",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/html",
    "text/x-python",
    "text/javascript",
    "text/typescript",
    "text/x-java-source",
    "text/x-c++src",
    "text/x-go",
    "text/x-rust",
}

# Global instances
metrics = PrometheusMetrics()
rate_limiter = RateLimiter(requests_per_minute=100)

# Create FastAPI app
app = FastAPI(
    title="RAG Pipeline API",
    description="Production-grade Retrieval-Augmented Generation pipeline",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    """Initialize pipeline on startup."""
    pipeline = get_pipeline()
    stats = pipeline.get_stats()
    print(f"RAG Pipeline initialized. Stats: {stats}")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    start = time.time()
    try:
        result = {
            "status": "healthy",
            "version": "0.1.0",
            "vector_store": settings.vector_store_type,
        }
        metrics.record_request("/health", time.time() - start, 200)
        return result
    except Exception:
        metrics.record_request("/health", time.time() - start, 500)
        raise


@app.post("/upload", response_model=DocumentIngestionResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document.

    Supported formats:
    - PDF (.pdf)
    - Text (.txt, .md, .markdown)
    - Code (.py, .js, .ts, .java, .cpp, .go, .rs)
    - Documents (.docx)
    - Web (.html)
    - Data (.json)
    """
    start = time.time()
    status_code = 200

    try:
        # Rate limiting
        if not rate_limiter.is_allowed("upload"):
            status_code = 429
            metrics.record_request("/upload", time.time() - start, status_code)
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Maximum 100 requests per minute."
            )

        if not file.filename:
            status_code = 400
            raise HTTPException(status_code=400, detail="No filename provided")

        # MIME type validation
        file_ext = Path(file.filename).suffix.lower()

        supported_extensions = {
            ".pdf", ".txt", ".md", ".markdown", ".docx", ".html", ".json",
            ".py", ".js", ".ts", ".java", ".cpp", ".go", ".rs"
        }

        if file_ext not in supported_extensions:
            status_code = 400
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Supported: {', '.join(supported_extensions)}"
            )

        # Check file size
        file_size_mb = 0
        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)

        if file_size_mb > settings.max_file_size_mb:
            status_code = 413
            raise HTTPException(
                status_code=413,
                detail=f"File size exceeds {settings.max_file_size_mb}MB limit"
            )

        # Save file temporarily
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = settings.upload_dir / file.filename

        try:
            with open(file_path, "wb") as f:
                f.write(content)

            # Ingest document
            pipeline = get_pipeline()
            result = pipeline.ingest_document(file_path)

            if result["status"] == "error":
                status_code = 400
                return DocumentIngestionResponse(
                    document_id="",
                    file_name=file.filename,
                    chunk_count=0,
                    status="error",
                    error=result.get("error", "Ingestion failed")
                )

            metrics.record_upload(len(content))
            status_code = 200
            return DocumentIngestionResponse(**result)

        except DocumentIngestionResponse as e:
            raise
        except HTTPException:
            raise
        except Exception as e:
            status_code = 500
            error_msg = str(e)
            return DocumentIngestionResponse(
                document_id="",
                file_name=file.filename,
                chunk_count=0,
                status="error",
                error=f"Upload failed: {error_msg}"
            )
        finally:
            # Clean up temporary file
            if file_path.exists():
                try:
                    file_path.unlink()
                except:
                    pass

    except HTTPException:
        metrics.record_request("/upload", time.time() - start, status_code)
        raise
    except Exception as e:
        status_code = 500
        metrics.record_request("/upload", time.time() - start, status_code)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Query documents for relevant chunks with support for multiple search modes.

    Search modes:
    - semantic: Dense vector similarity only
    - keyword: BM25 keyword search only
    - hybrid: Combines semantic + keyword with reciprocal rank fusion + cross-encoder reranking

    Returns top-K most relevant chunks with:
    - Chunk content (text)
    - Metadata (source file, page number, section headers, etc.)
    - Similarity score (0-1)
    """
    start = time.time()
    status_code = 200

    try:
        # Rate limiting
        if not rate_limiter.is_allowed("query"):
            status_code = 429
            metrics.record_request("/query", time.time() - start, status_code)
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Maximum 100 requests per minute."
            )

        if not request.query or len(request.query.strip()) == 0:
            status_code = 400
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        # Validate search mode
        valid_modes = ["semantic", "keyword", "hybrid"]
        mode = request.mode or "semantic"
        if mode not in valid_modes:
            status_code = 400
            raise HTTPException(
                status_code=400,
                detail=f"Invalid search mode '{mode}'. Must be one of: {', '.join(valid_modes)}"
            )

        try:
            pipeline = get_pipeline()
            result = pipeline.query(request.query, request.top_k, mode=mode)

            if result["status"] == "error":
                status_code = 500
                raise HTTPException(status_code=500, detail=result.get("error", "Query failed"))

            metrics.record_query()
            status_code = 200
            return QueryResponse(**result)

        except HTTPException:
            raise
        except Exception as e:
            status_code = 500
            raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

    finally:
        metrics.record_request("/query", time.time() - start, status_code)


@app.get("/stats")
async def get_stats():
    """Get pipeline statistics."""
    start = time.time()
    try:
        pipeline = get_pipeline()
        result = pipeline.get_stats()
        metrics.record_request("/stats", time.time() - start, 200)
        return result
    except Exception as e:
        metrics.record_request("/stats", time.time() - start, 500)
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its chunks."""
    start = time.time()
    status_code = 200

    try:
        pipeline = get_pipeline()
        result = pipeline.delete_document(document_id)

        if result["status"] == "error":
            status_code = 500
            raise HTTPException(status_code=500, detail=result.get("error", "Deletion failed"))

        return result

    except HTTPException:
        status_code = 500
        raise
    except Exception as e:
        status_code = 500
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")
    finally:
        metrics.record_request("/documents", time.time() - start, status_code)


@app.post("/clear-all")
async def clear_all_chunks():
    """Clear all documents and chunks from the vector store."""
    start = time.time()
    status_code = 200

    try:
        pipeline = get_pipeline()
        result = pipeline.vector_store.clear_all()
        return {
            "status": "success",
            "message": "All chunks cleared successfully",
            "result": result
        }
    except Exception as e:
        status_code = 500
        raise HTTPException(status_code=500, detail=f"Failed to clear chunks: {str(e)}")
    finally:
        metrics.record_request("/clear-all", time.time() - start, status_code)


@app.get("/metrics")
async def get_metrics():
    """Get Prometheus-format metrics."""
    return PlainTextResponse(metrics.get_metrics())


@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "message": "RAG Pipeline API",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "POST /upload": "Upload and ingest a document",
            "POST /query": "Query for relevant chunks",
            "GET /stats": "Get pipeline statistics",
            "DELETE /documents/{document_id}": "Delete a document",
            "GET /health": "Health check",
            "GET /metrics": "Prometheus metrics",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
    )
