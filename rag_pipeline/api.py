"""FastAPI REST API for RAG pipeline."""

import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from rag_pipeline.config import settings
from rag_pipeline.generation import generate_answer, get_llm_client
from rag_pipeline.pipeline import get_pipeline
from rag_pipeline.security.auth import TenantContext, require_api_key
from rag_pipeline.security.tenancy import reset_tenant_context, set_tenant_context


# Pydantic models for request/response
class QueryRequest(BaseModel):
    """Query request model."""

    query: str
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    mode: Optional[str] = "semantic"  # 'semantic', 'keyword', or 'hybrid'
    rerank: bool = False  # Apply cross-encoder reranking


class QueryResponse(BaseModel):
    """Query response model."""

    query: str
    result_count: int
    results: list[dict]
    status: str
    mode: str = "semantic"
    reranked: bool = False


class AnswerRequest(BaseModel):
    """Answer request model."""

    query: str
    top_k: int = Field(default=5, ge=1, le=100)
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"query": "What is machine learning?", "top_k": 5}
        }
    )


class CitationModel(BaseModel):
    """Citation in answer."""

    chunk_id: str
    chunk_index: int
    source_file: Optional[str] = None
    page_number: Optional[int] = None


class AnswerResponse(BaseModel):
    """Answer response model."""

    query: str
    answer: str
    citations: list[CitationModel] = Field(default_factory=list)
    confidence: float
    abstained: bool
    status: str

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "What is machine learning?",
                "answer": "Machine learning is a subset of AI [chunk_1].",
                "citations": [
                    {
                        "chunk_id": "chunk_1",
                        "chunk_index": 1,
                        "source_file": "ml.pdf",
                        "page_number": 3,
                    }
                ],
                "confidence": 0.92,
                "abstained": False,
                "status": "success",
            }
        }
    )


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


async def request_tenant(
    context: TenantContext = Depends(require_api_key),
):
    """Authenticate the request and scope tenant state for its full lifecycle."""
    token = set_tenant_context(context)
    try:
        yield context
    finally:
        reset_tenant_context(token)


def _rate_limit_key(request: Request, tenant: TenantContext, endpoint: str) -> str:
    client_host = request.client.host if request.client else "unknown"
    identity = tenant.tenant_id if tenant.tenant_id != "public" else client_host
    return f"{endpoint}:{identity}"


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the shared pipeline for the lifetime of the API process."""
    pipeline = get_pipeline()
    stats = pipeline.get_stats()
    print(f"RAG Pipeline initialized. Stats: {stats}")
    yield


# Create FastAPI app
app = FastAPI(
    title="RAG Pipeline API",
    description="Production-grade Retrieval-Augmented Generation pipeline",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
async def upload_document(
    http_request: Request,
    file: UploadFile = File(...),
    tenant: TenantContext = Depends(request_tenant),
):
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
        if not rate_limiter.is_allowed(_rate_limit_key(http_request, tenant, "upload")):
            status_code = 429
            metrics.record_request("/upload", time.time() - start, status_code)
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Maximum 100 requests per minute."
            )

        if not file.filename:
            status_code = 400
            raise HTTPException(status_code=400, detail="No filename provided")

        # Never use the client filename as a filesystem path. Keep only its
        # basename for display and derive the extension for parser selection.
        display_name = file.filename.replace("\\", "/").rsplit("/", 1)[-1]
        file_ext = Path(display_name).suffix.lower()

        supported_extensions = {
            ".pdf", ".txt", ".md", ".markdown", ".docx", ".html", ".json",
            ".py", ".js", ".ts", ".java", ".cpp", ".go", ".rs"
        }

        if file_ext not in supported_extensions:
            status_code = 400
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type: {file_ext}. "
                    f"Supported: {', '.join(supported_extensions)}"
                )
            )

        # Stream to a server-generated path and enforce the size limit as bytes
        # arrive, so a client cannot choose the path or force a full-memory read.
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        file_path = settings.upload_dir / f"upload_{uuid.uuid4().hex}{file_ext}"
        uploaded_bytes = 0

        try:
            with file_path.open("xb") as destination:
                while chunk := await file.read(1024 * 1024):
                    uploaded_bytes += len(chunk)
                    if uploaded_bytes > settings.max_file_size_mb * 1024 * 1024:
                        status_code = 413
                        raise HTTPException(
                            status_code=413,
                            detail=f"File size exceeds {settings.max_file_size_mb}MB limit",
                        )
                    destination.write(chunk)

            # Ingest document
            pipeline = get_pipeline()
            result = pipeline.ingest_document(
                file_path,
                tenant_id=tenant.tenant_id,
                source_name=display_name,
            )

            if result["status"] == "error":
                status_code = 400
                return DocumentIngestionResponse(
                    document_id="",
                    file_name=display_name,
                    chunk_count=0,
                    status="error",
                    error=result.get("error", "Ingestion failed")
                )

            metrics.record_upload(uploaded_bytes)
            status_code = 200
            result["file_name"] = display_name
            return DocumentIngestionResponse(**result)

        except HTTPException:
            raise
        except Exception as e:
            status_code = 500
            error_msg = str(e)
            return DocumentIngestionResponse(
                document_id="",
                file_name=display_name,
                chunk_count=0,
                status="error",
                error=f"Upload failed: {error_msg}"
            )
        finally:
            # Clean up temporary file
            try:
                file_path.unlink(missing_ok=True)
            except OSError:
                pass

    except HTTPException:
        metrics.record_request("/upload", time.time() - start, status_code)
        raise
    except Exception as e:
        status_code = 500
        metrics.record_request("/upload", time.time() - start, status_code)
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    http_request: Request,
    tenant: TenantContext = Depends(request_tenant),
):
    """
    Query documents for relevant chunks with support for multiple search modes.

    Search modes:
    - semantic: Dense vector similarity only
    - keyword: BM25 keyword search only
    - hybrid: Combines semantic + BM25 results with reciprocal rank fusion

    Set rerank=true to rescore an expanded candidate pool with the cross-encoder
    and use MMR to reduce redundant results.

    Returns top-K most relevant chunks with:
    - Chunk content (text)
    - Metadata (source file, page number, section headers, etc.)
- Ranking score (semantic similarity, normalized BM25, or RRF depending on mode)
    """
    start = time.time()
    status_code = 200

    try:
        # Rate limiting
        if not rate_limiter.is_allowed(_rate_limit_key(http_request, tenant, "query")):
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
            result = pipeline.query(
                request.query,
                request.top_k,
                mode=mode,
                rerank=request.rerank,
                tenant_id=tenant.tenant_id,
            )

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


@app.post("/answer", response_model=AnswerResponse)
async def answer_question(
    request: AnswerRequest,
    http_request: Request,
    tenant: TenantContext = Depends(request_tenant),
):
    """
    Generate an answer to a query using retrieved documents.

    Retrieves relevant chunks, then generates a natural language answer with:
    - Answer text citing source chunks
    - Citations with source metadata
    - Confidence score based on retrieval quality
    - Abstention flag if confidence is too low

    Returns 200 with answer or 400/500 on error.
    """
    start = time.time()
    status_code = 200

    try:
        # Rate limiting
        if not rate_limiter.is_allowed(_rate_limit_key(http_request, tenant, "answer")):
            status_code = 429
            metrics.record_request("/answer", time.time() - start, status_code)
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Maximum 100 requests per minute."
            )

        if not request.query or len(request.query.strip()) == 0:
            status_code = 400
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        try:
            # Retrieve relevant chunks
            pipeline = get_pipeline()
            retrieval_result = pipeline.query(
                request.query, request.top_k, mode="semantic", tenant_id=tenant.tenant_id
            )

            if retrieval_result["status"] == "error":
                status_code = 500
                raise HTTPException(
                    status_code=500,
                    detail=retrieval_result.get("error", "Retrieval failed")
                )

            # Convert retrieval results to chunk format for generation
            chunks = []
            for result in retrieval_result.get("results", []):
                chunks.append({
                    "content": result.get("content", ""),
                    "metadata": result.get("metadata", {}),
                    "score": result.get("similarity_score", 0.0),
                })

            # Generate answer
            llm_provider = settings.llm_provider or "fake"
            llm_client = get_llm_client(llm_provider)
            abstain_threshold = settings.abstain_threshold or 0.35

            answer_result = await generate_answer(
                request.query,
                chunks,
                llm_client,
                abstain_threshold=abstain_threshold,
            )

            # Convert citations
            citations = [
                CitationModel(
                    chunk_id=c.chunk_id,
                    chunk_index=c.chunk_index,
                    source_file=c.source_file,
                    page_number=c.page_number,
                )
                for c in answer_result.citations
            ]

            metrics.record_query()
            status_code = 200
            return AnswerResponse(
                query=request.query,
                answer=answer_result.answer,
                citations=citations,
                confidence=answer_result.confidence,
                abstained=answer_result.abstained,
                status="success",
            )

        except HTTPException:
            raise
        except Exception as e:
            status_code = 500
            raise HTTPException(status_code=500, detail=f"Answer generation failed: {str(e)}")

    finally:
        metrics.record_request("/answer", time.time() - start, status_code)


@app.get("/stats")
async def get_stats(tenant: TenantContext = Depends(request_tenant)):
    """Get pipeline statistics."""
    start = time.time()
    try:
        pipeline = get_pipeline()
        result = pipeline.get_stats(tenant_id=tenant.tenant_id)
        metrics.record_request("/stats", time.time() - start, 200)
        return result
    except Exception as e:
        metrics.record_request("/stats", time.time() - start, 500)
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    tenant: TenantContext = Depends(request_tenant),
):
    """Delete a document and its chunks."""
    start = time.time()
    status_code = 200

    try:
        pipeline = get_pipeline()
        result = pipeline.delete_document(document_id, tenant_id=tenant.tenant_id)

        if result["status"] == "not_found":
            status_code = 404
            raise HTTPException(status_code=404, detail="Document not found")
        if result["status"] == "error":
            status_code = 500
            raise HTTPException(status_code=500, detail=result.get("error", "Deletion failed"))

        return result

    except HTTPException as exc:
        status_code = exc.status_code
        raise
    except Exception as e:
        status_code = 500
        raise HTTPException(status_code=500, detail=f"Deletion failed: {str(e)}")
    finally:
        metrics.record_request("/documents", time.time() - start, status_code)


@app.post("/clear-all")
async def clear_all_chunks(tenant: TenantContext = Depends(request_tenant)):
    """Clear only the authenticated tenant's documents and chunks."""
    start = time.time()
    status_code = 200

    try:
        pipeline = get_pipeline()
        result = pipeline.clear_tenant(tenant_id=tenant.tenant_id)
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
async def get_metrics(tenant: TenantContext = Depends(request_tenant)):
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
            "POST /answer": "Generate an answer with validated source citations",
            "GET /stats": "Get pipeline statistics",
            "DELETE /documents/{document_id}": "Delete a document",
            "POST /clear-all": "Clear this tenant's documents",
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
