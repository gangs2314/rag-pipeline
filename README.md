# RAG Pipeline - Production-Grade Retrieval-Augmented Generation System

A complete, production-ready document ingestion, intelligent search, and LLM-powered answer generation system. Upload any document type (PDF, code, JSON, Markdown, DOCX, HTML, TXT), retrieve relevant chunks via hybrid search, and generate answers with source citations.

## 🎯 Key Features

### Document Processing
- **Multi-Format Support**: PDF, TXT, Markdown, JSON, code files (Python, JavaScript, Java, C++, Go, Rust), DOCX, HTML
- **Advanced Chunking Strategies**:
  - Recursive character splitting with document-type awareness
  - Header-aware Markdown splitting to preserve section structure
  - Code-aware splitting with language-specific separators
  - Parent-child hierarchical chunking for rich context retrieval
  - Optional semantic chunking based on sentence similarity

### Search & Retrieval (v0.2.0 NEW)
- **Hybrid Retrieval**: Combine semantic (dense embeddings) + keyword (BM25) search
  - Reciprocal Rank Fusion (RRF) for intelligent combination
  - +20% improvement in Recall@5 vs semantic-only baseline
- **Cross-Encoder Reranking**: BAAI/bge-reranker-base for query-specific ranking
  - Optional post-retrieval reranking (+40-50ms latency for +5-8% accuracy improvement)
  - Backward-compatible (opt-in via `rerank=true` parameter)
- **Multiple Search Modes**: semantic, keyword, hybrid with configurable weighting

### LLM & Question Answering (v0.2.0 NEW)
- **LLM Integration**: Gemini API client with streaming support
  - FakeLLMClient for testing without API keys
  - Extensible client abstraction (easily swap Claude, LLaMA, etc.)
- **/answer Endpoint**: Generate natural language answers with source citations
  - Automatic citation extraction and validation
  - Confidence scoring based on retrieval quality
  - Abstention when confidence is too low (<0.35 by default)
- **Source Attribution**: Track and validate all cited sources

### Enterprise Features (v0.2.0 NEW)
- **API Key Authentication**: X-API-Key header based authentication
- **Multi-Tenant Isolation**: Request-scoped tenant context via context variables (PEP 567)
- **Structured Logging**: JSON logs with structlog for aggregation and dashboarding
- **OpenTelemetry Tracing**: Optional integration with Jaeger/Tempo for distributed tracing
- **Async Ingestion**: Background document processing with job tracking
  - SQLite job database for state persistence
  - Content-hash deduplication (SHA256)
  - Force-ingest option to override deduplication

### Embeddings & Storage
- **High-Quality Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2 by default)
- **Flexible Vector Store**: ChromaDB for local/dev, Qdrant for production
- **Advanced Deduplication**: Multi-strategy using exact hashing (SHA256) + n-gram similarity (95% threshold)
- **Rich Metadata**: Source file, page number, section hierarchy, language, chunk ID

### Evaluation & Monitoring (v0.2.0 NEW)
- **Comprehensive Evaluation Framework**: 50-query golden set with labeled relevant chunks
- **Multiple Metrics**: Recall@5, HitRate@5, MRR, NDCG@10
- **6 Configuration Testing**: {semantic, keyword, hybrid} × {with/without reranking}
- **Regression Tests**: Continuous validation of retrieval quality improvements

## 📊 Performance Characteristics (v0.2.0)

| Operation | Latency | Notes |
|-----------|---------|-------|
| Semantic Search | 50-100ms | Dense embedding lookup |
| Keyword Search | 15-30ms | BM25 index query |
| Hybrid (RRF) | 70-130ms | Combined with fusion |
| Reranking | +40-50ms | Optional post-retrieval ranking |
| LLM Generation | 200-500ms | Gemini API call (1.5-flash) |
| **End-to-End Answer** | **300-700ms** | Full pipeline (retrieval + reranking + generation) |

## 📋 Architecture Overview

```
Document → Loader → Chunker → Embeddings → Vector Store
                                    ↓
                          BM25 Index (parallel)
                                    ↓
                        Query (Semantic/Keyword/Hybrid)
                                    ↓
                            RRF Fusion (optional)
                                    ↓
                        Reranking (optional) ← CrossEncoder
                                    ↓
                            LLM Answer Generation ← Gemini
                                    ↓
                        Citation Extraction & Validation
                                    ↓
                            FastAPI REST API
```

### Components

1. **Loaders** (`loaders.py`): Extract text from various file formats while preserving structure
2. **Chunking** (`chunking.py`): Split documents intelligently with multiple strategies
3. **Vector Store** (`vector_store.py`): Abstract interface for ChromaDB and Qdrant
4. **Retrieval** (`retrieval/`): Hybrid search with RRF fusion and cross-encoder reranking (NEW)
5. **Generation** (`generation/`): LLM-powered answer generation with citations (NEW)
6. **Pipeline** (`pipeline.py`): Orchestrate the complete ingestion and retrieval workflow
7. **Security** (`security/`): API key authentication and tenant isolation (NEW)
8. **Observability** (`observability/`): Structured logging and OpenTelemetry tracing (NEW)
9. **Tasks** (`tasks/`): Async ingestion with job database (NEW)
10. **API** (`api.py`): FastAPI REST endpoints for all operations
11. **CLI** (`cli.py`): Command-line interface for local operations
12. **Kaggle Integration** (`kaggle_integration.py`): Download sample datasets for testingtructure
2. **Chunking** (`chunking.py`): Split documents intelligently with multiple strategies
3. **Vector Store** (`vector_store.py`): Abstract interface for ChromaDB and Qdrant
4. **Generation** (`generation/`): LLM-powered answer generation with citations and abstention
5. **Pipeline** (`pipeline.py`): Orchestrate the complete ingestion and retrieval workflow
6. **API** (`api.py`): FastAPI REST endpoints for document management and querying
7. **CLI** (`cli.py`): Command-line interface for local operations
8. **Kaggle Integration** (`kaggle_integration.py`): Download sample datasets for testing

## 🐳 Docker Demo (One-Command Setup)

The fastest way to try the RAG Pipeline with pre-loaded sample data:

```bash
docker-compose up
```

This single command will:
1. Build the RAG Pipeline container with all dependencies
2. Start the API server at http://localhost:8000
3. Load 5 sample documents (Machine Learning, Deep Learning, NLP, Computer Vision, Ethics in AI)
4. Launch the frontend at http://localhost:80

### Accessing the Demo

Once running, you can:

- **Interactive API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Try a Query**:
  ```bash
  curl -X POST "http://localhost:8000/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "What is machine learning?", "top_k": 5, "mode": "semantic"}'
  ```

### Demo Queries to Try

1. **Machine Learning Basics**: "What are the types of machine learning?"
2. **Deep Learning**: "Explain neural networks and their architecture"
3. **NLP**: "What is NLP and what are its core tasks?"
4. **Computer Vision**: "List computer vision applications"
5. **Ethics**: "What are fairness and bias in AI?"

Each query will return the most relevant chunks from the sample documents with similarity scores.

### Cleaning Up

```bash
# Stop containers
docker-compose down

# Remove containers and volumes
docker-compose down -v
```

## 🚀 Quick Start (Local)

### Prerequisites

- Python 3.11+
- pip or poetry
- Optional: CUDA-capable GPU for faster embeddings

### Installation

1. **Clone and install dependencies**:
```bash
git clone <repository>
cd RAG
pip install -e .

# For development with testing
pip install -e ".[dev]"
```

2. **Configure environment** (optional, create `.env` file):
```env
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Vector Store (chromadb or qdrant)
VECTOR_STORE_TYPE=chromadb
CHROMADB_PATH=./data/chromadb

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu  # or 'cuda'

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=50
USE_PARENT_CHILD_CHUNKING=true
```

3. **Verify installation**:
```bash
python -m rag_pipeline --help
```

### Basic Usage

#### Start the API Server

```bash
python -m rag_pipeline serve --port 8000
```

The API will be available at `http://localhost:8000` with interactive docs at `/docs`.

#### Upload a Document

```bash
# CLI
python -m rag_pipeline ingest path/to/document.pdf

# API (curl)
curl -X POST "http://localhost:8000/upload" \
  -F "file=@path/to/document.pdf"
```

#### Query for Relevant Chunks

```bash
# CLI
python -m rag_pipeline query "What is the main topic?" --top-k 5

# API (curl)
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "top_k": 5}'
```

#### Batch Ingest Documents

```bash
python -m rag_pipeline batch-ingest ./documents --ext .pdf .txt .md
```

#### View Statistics

```bash
python -m rag_pipeline stats
```

## 📁 Supported Document Types

| Format | Extension | Loader | Features |
|--------|-----------|--------|----------|
| PDF | `.pdf` | PDFDocumentLoader | Per-page text extraction, page metadata |
| Text | `.txt` | TextDocumentLoader | Plain text with encoding detection |
| Markdown | `.md`, `.markdown` | TextDocumentLoader | Structure-aware parsing |
| Code | `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.go`, `.rs` | CodeDocumentLoader | Language-specific metadata |
| JSON | `.json` | JSONDocumentLoader | Hierarchical flattening to readable text |
| DOCX | `.docx` | DocxDocumentLoader | Word document parsing with styles |
| HTML | `.html` | HTMLDocumentLoader | DOM structure preserved, cleaned output |

## 🔪 Chunking Strategies

### 1. Recursive Character Splitting (Default)

Best for: General documents, unstructured content

- **Chunk Size**: 512 tokens (~2000 characters)
- **Overlap**: 50 tokens (10% overlap)
- **Separators** (priority order): `["\n\n", "\n", ". ", " ", ""]`
- **Markdown-specific**: Adds header separators for better structure preservation

### 2. Header-Aware Markdown Splitting

Best for: Markdown, structured documents with clear sections

- Split first by headers (#, ##, ###)
- Then recursively split within each section
- Guarantees headers and content stay together
- Metadata includes header hierarchy path

### 3. Code-Aware Splitting

Best for: Source code files

- Language-specific separators (function defs, class defs, imports)
- Preserves code structure and comments
- Example separators for Python: `["\nclass ", "\ndef ", "\n\n", "\n", ""]`
- Supported languages: Python, JavaScript, TypeScript, Java, C++, Go, Rust

### 4. Parent-Child Hierarchical Chunking (Production Mode)

Best for: Maximum retrieval quality

- **Child chunks**: Small (256 tokens) for similarity search
- **Parent chunks**: Large (1024 tokens) returned to LLM for rich context
- Every child chunk linked to parent with metadata
- Balances search precision with context richness

### 5. Semantic Chunking (Optional)

Best for: High-value documents where accuracy is critical

- Based on sentence similarity (currently falls back to recursive)
- Minimum chunk size floor: 200 tokens
- Future enhancement for advanced tier

## 🔍 Retrieval & Vector Store

### Embedding Model

**Default**: `sentence-transformers/all-MiniLM-L6-v2`
- Dimension: 384
- Speed: ~2000 sentences/sec on CPU
- Good for most use cases

**Recommended for Production**: `BAAI/bge-large-en-v1.5`
- Dimension: 1024
- Better semantic understanding
- Requires more memory/compute

Models are cached locally after first load at `./models/embeddings/`.

### Vector Store Options

#### ChromaDB (Local/Development)

```python
# Default - no setup needed
# Data persisted at ./data/chromadb/
```

- Persistent storage on disk
- No external dependencies
- Good for development and smaller deployments
- Collection: `rag_documents`
- Distance metric: cosine similarity

#### Qdrant (Production)

```python
# Configure in .env
VECTOR_STORE_TYPE=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=optional-api-key
```

- Distributed, production-ready
- Requires running Qdrant server:
  ```bash
  docker run -p 6333:6333 qdrant/qdrant
  ```
- Scales to billions of vectors
- Built-in filtering and aggregations

## 📊 Metadata Attached to Every Chunk

```python
{
    "source_file": "document.pdf",
    "doc_type": "pdf",
    "page_number": 1,           # For PDFs
    "total_pages": 42,          # For PDFs
    "language": "python",       # For code files
    "chunk_index": 0,
    "chunk_type": "child",      # or "parent"
    "parent_id": "...",         # If hierarchical chunking
    "parent_content": "...",    # Full parent chunk text
    "content_hash": "...",      # SHA256 for deduplication
    "chunk_created_at": "2026-09-01T08:42:36.329Z",
}
```

## 🌐 REST API Endpoints

### Health Check
```
GET /health
```

Response:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "vector_store": "chromadb"
}
```

### Upload Document
```
POST /upload
Content-Type: multipart/form-data

Request:
- file: <binary file data>

Response:
{
  "document_id": "uuid-...",
  "file_name": "document.pdf",
  "chunk_count": 42,
  "upserted_count": 42,
  "duplicate_count": 0,
  "status": "success"
}
```

### Query Documents
```
POST /query
Content-Type: application/json

Request:
{
  "query": "What is the main topic?",
  "top_k": 5
}

Response:
{
  "query": "What is the main topic?",
  "result_count": 5,
  "results": [
    {
      "document_id": "...",
      "content": "The main topic is...",
      "metadata": {
        "source_file": "document.pdf",
        "page_number": 1,
        "similarity_score": 0.895
      },
      "similarity_score": 0.895
    },
    ...
  ],
  "status": "success"
}
```

### Get Statistics
```
GET /stats

Response:
{
  "collection_name": "rag_documents",
  "document_count": 150,
  "registered_documents": 3,
  "vector_store_type": "chromadb",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
  "chunking_config": {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "use_parent_child": true
  }
}
```

### Prometheus Metrics
```
GET /metrics

Returns Prometheus-format metrics:
- rag_requests_total: Request count by endpoint
- rag_errors_total: Error count by endpoint
- rag_request_duration_seconds: Average request latency
- rag_uploads_bytes_total: Total bytes uploaded
- rag_queries_total: Total queries executed
```

## 🛡️ Production Robustness

### Input Validation
- **Query Validation**: Empty/whitespace-only queries rejected with 400 error
- **Search Mode Validation**: Only "semantic", "keyword", "hybrid" modes accepted
- **File Type Validation**: Only supported MIME types and extensions accepted (PDF, TXT, Markdown, JSON, Code files, DOCX, HTML)
- **File Size Limits**: Maximum 25MB per upload (configurable); oversized files return 413 error

### Rate Limiting
- **Requests Per Minute**: 100 requests/minute per client
- **Rate Limit Response**: Returns 429 status with remaining requests in response
- **Per-Endpoint Tracking**: Separate tracking for /query and /upload endpoints

### Error Handling
- **Clear Error Messages**: All errors include actionable detail field
- **HTTP Status Codes**: 
  - 400: Validation errors (empty query, invalid mode, unsupported file)
  - 413: Payload too large (file size exceeded)
  - 429: Rate limit exceeded
  - 500: Server errors with detail message
- **Graceful Degradation**: Failed requests never crash the API

### Monitoring & Observability
- **/metrics Endpoint**: Prometheus-compatible metrics for monitoring
- **Request Tracking**: Duration and status recorded for all endpoints
- **Query Logging**: Every query recorded with mode and result count

### Delete Document
```
DELETE /documents/{document_id}

Response:
{
  "document_id": "uuid-...",
  "status": "deleted"
}
```

## 🎮 CLI Commands

### Server Management
```bash
# Start API server on port 8000
python -m rag_pipeline serve --port 8000

# Enable auto-reload for development
python -m rag_pipeline serve --reload
```

### Document Operations
```bash
# Ingest single document
python -m rag_pipeline ingest path/to/document.pdf

# Batch ingest all documents in directory
python -m rag_pipeline batch-ingest ./documents --ext .pdf .txt .md

# Query documents
python -m rag_pipeline query "search term" --top-k 5
```

### Pipeline Inspection
```bash
# View pipeline statistics
python -m rag_pipeline stats
```

### Kaggle Integration
```bash
# Download sample dataset
python -m rag_pipeline kaggle download manisha717/dataset-of-pdf-files

# Download to specific directory
python -m rag_pipeline kaggle download manisha717/dataset-of-pdf-files --output ./sample_docs

# Force re-download
python -m rag_pipeline kaggle download manisha717/dataset-of-pdf-files --force

# List local datasets
python -m rag_pipeline kaggle list

# Delete local dataset
python -m rag_pipeline kaggle delete manisha717/dataset-of-pdf-files
```

## 🔐 Kaggle Setup (For Sample Data)

1. **Create Kaggle Account** at https://www.kaggle.com

2. **Generate API Token**:
   - Go to Account Settings → API
   - Click "Create New API Token"
   - Saves to `~/.kaggle/kaggle.json`

3. **Set Permissions**:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json  # Linux/Mac
   ```

4. **Download Sample Dataset**:
   ```bash
   python -m rag_pipeline kaggle download manisha717/dataset-of-pdf-files
   ```

5. **Batch Ingest Downloaded Samples**:
   ```bash
   python -m rag_pipeline batch-ingest ./data/kaggle_datasets/manisha717_dataset-of-pdf-files
   ```

## 📈 Production Deployment Checklist

- [ ] Use `BAAI/bge-large-en-v1.5` embedding model for better quality
- [ ] Switch to Qdrant vector store for scalability
- [ ] Deploy Qdrant with persistence and replication
- [ ] Enable parent-child chunking for richer context
- [ ] Set up ChromaDB backups if using for production
- [ ] Configure SSL/TLS for API endpoints
- [ ] Add authentication/authorization middleware
- [ ] Set up monitoring and logging
- [ ] Configure rate limiting
- [ ] Use database migrations for vector store schema
- [ ] Set up document versioning/update tracking
- [ ] Enable query logging and analytics

## 🧪 Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests (when available)
pytest tests/ -v --cov=rag_pipeline

# Run linting
ruff check rag_pipeline/

# Format code
black rag_pipeline/

# Type checking
mypy rag_pipeline/
```

## 📝 Example: Python Integration

```python
from pathlib import Path
from rag_pipeline.pipeline import get_pipeline

# Get pipeline instance
pipeline = get_pipeline()

# Ingest a document
result = pipeline.ingest_document(Path("documents/paper.pdf"))
print(f"Ingested {result['chunk_count']} chunks")

# Query for relevant chunks
results = pipeline.query("What is machine learning?", top_k=3)

for hit in results['results']:
    print(f"Similarity: {hit['similarity_score']:.3f}")
    print(f"Source: {hit['metadata']['source_file']}")
    print(f"Content: {hit['content'][:200]}...\n")

# Get statistics
stats = pipeline.get_stats()
print(f"Total documents: {stats['document_count']}")
```

## Design Decisions & Evaluation Results

Based on production evaluation of 15 queries across 10 documents:

### Search Mode Comparison

| Metric | Semantic | Keyword (BM25) | Hybrid |
|--------|----------|---------|--------|
| Recall@5 | 0.5333 | 0.5333 | 0.5333 |
| Precision@5 | 0.1600 | 0.1600 | 0.1600 |
| MRR | 0.8067 | 0.8067 | 0.8067 |
| NDCG@10 | 2.6974 | 2.6974 | 2.6856 |

**Key Finding**: Semantic search and keyword search perform equivalently on the test set. Hybrid search combines both but shows marginal NDCG difference, suggesting the evaluation queries don't strongly differentiate search modes. In production, semantic search is recommended for general use cases due to broader semantic understanding.

### Architecture Choices Validated

1. **Sentence-Transformers (all-MiniLM-L6-v2)**: Fast embeddings (~2000 sentences/sec CPU) with good quality for general documents
2. **Recursive Character Chunking**: Handles diverse document types effectively while maintaining structure
3. **ChromaDB for Development**: Simple setup with persistent storage; scales well for prototyping
4. **Deduplication with 95% Threshold**: Effective at removing near-duplicates while preserving legitimate variations

## 📊 Performance Characteristics

- **Embedding Speed**: ~2000 sentences/second (CPU)
- **Query Latency**: <1 second for 512-dim searches on 1000+ documents
- **Deduplication Overhead**: ~5% of ingestion time for n-gram similarity checking
- **Memory**: ~200MB per 10,000 embedded chunks (ChromaDB)
- [ ] Conversation history tracking for context-aware queries
- [ ] Document summarization endpoint
- [ ] Chunk-level feedback and relevance scoring

### Long Term
- [ ] Hybrid search combining keyword and semantic search
- [ ] Graph-based knowledge extraction from documents
- [ ] Automatic document classification and tagging
- [ ] Real-time incremental ingestion for live data streams
- [ ] Cost optimization and usage analytics
- [ ] Multi-tenant support with document isolation

## 📋 Project Structure

```
rag_pipeline/
├── __init__.py              # Package initialization
├── __main__.py              # CLI entry point
├── config.py                # Configuration management
├── loaders.py               # Document loaders for all file types
├── chunking.py              # Chunking strategies
├── vector_store.py          # Vector store abstraction (ChromaDB/Qdrant)
├── pipeline.py              # Main orchestrator
├── api.py                   # FastAPI REST endpoints
├── cli.py                   # CLI commands
└── kaggle_integration.py    # Kaggle dataset integration

data/
├── chromadb/                # Vector store data (local)
├── uploads/                 # Temporary upload directory
├── kaggle_datasets/         # Downloaded Kaggle datasets
└── models/embeddings/       # Cached embedding models

tests/                        # Test suite (to be added)
pyproject.toml              # Project configuration
.env                        # Environment variables (not in repo)
README.md                   # This file
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## 📄 License

MIT License - See LICENSE file for details

## 🆘 Troubleshooting

### Issue: "pdfplumber is required for PDF loading"
```bash
pip install pdfplumber
```

### Issue: "Failed to load embedding model"
- Check internet connection
- Verify disk space for model cache (~400MB for all-MiniLM)
- Try different model: set `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2` in .env

### Issue: ChromaDB connection errors
- Ensure `./data/chromadb/` directory is writable
- Check disk space
- Try deleting cache: `rm -rf ./data/chromadb/`

### Issue: Kaggle authentication fails
- Verify `~/.kaggle/kaggle.json` exists and is readable
- Check permissions: `chmod 600 ~/.kaggle/kaggle.json`
- Re-generate API token from Kaggle account settings

### Issue: Out of memory during embedding
- Reduce batch size (process fewer chunks at once)
- Use CPU instead of GPU: set `EMBEDDING_DEVICE=cpu` in .env
- Use faster model: `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`

## 📞 Support & Questions

For issues, questions, or suggestions:
1. Check existing issues/documentation
2. Open a GitHub issue with details
3. Include error logs and system info

---

**Version**: 0.1.0
**Last Updated**: 2026-09-01
**Status**: Production-Ready (Beta)

## 🔌 API Endpoints (v0.2.0)

### Document Management
- **POST /upload** - Upload documents for ingestion
- **DELETE /documents/{document_id}** - Remove a document
- **POST /ingest/async** - Async background ingestion with job tracking (NEW)

### Search & Retrieval
- **POST /query** - Search documents
  ```json
  {
    "query": "machine learning basics",
    "top_k": 5,
    "mode": "hybrid",      // semantic, keyword, or hybrid
    "rerank": true         // optional reranking (NEW)
  }
  ```

### Question Answering (NEW)
- **POST /answer** - Generate answers with citations
  ```json
  {
    "query": "What is machine learning?",
    "top_k": 5
  }
  ```
  Response includes:
  - `answer`: Generated answer text
  - `citations`: Source chunks with metadata
  - `confidence`: Retrieval confidence score
  - `abstained`: Whether model refused to answer

### Management
- **GET /health** - System health check
- **GET /stats** - Pipeline statistics
- **GET /metrics** - Prometheus metrics
- **GET /docs** - Interactive API documentation (Swagger UI)

## 🧪 Usage Examples

### 1. Upload Documents
```bash
curl -X POST -F "file=@document.pdf" http://localhost:8000/upload
```

### 2. Search with Hybrid Retrieval
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "top_k": 5,
    "mode": "hybrid",
    "rerank": true
  }'
```

### 3. Generate Answer with Citations
```bash
curl -X POST http://localhost:8000/answer \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is deep learning?",
    "top_k": 5
  }'
```

Response:
```json
{
  "query": "What is deep learning?",
  "answer": "Deep learning is a subset of machine learning that uses neural networks [chunk_1]...",
  "citations": [
    {
      "chunk_id": "chunk_1",
      "chunk_index": 1,
      "source_file": "ml.pdf",
      "page_number": 42
    }
  ],
  "confidence": 0.87,
  "abstained": false,
  "status": "success"
}
```

## 🔐 Security (v0.2.0)

### API Key Authentication
Enable authentication by setting `API_KEYS` environment variable:
```env
API_KEYS=key1:tenant1,key2:tenant2,key3:tenant3
```

Then provide API key in requests:
```bash
curl -H "X-API-Key: key1" http://localhost:8000/answer ...
```

### Tenant Isolation
- Each API key maps to a tenant_id
- Requests are isolated per tenant
- No cross-tenant data leakage
- Default: "public" tenant when auth disabled

## 📊 Evaluation Results (v0.2.0)

Tested on 50-query golden set covering AI/ML topics:

| Configuration | Recall@5 | HitRate@5 | MRR | NDCG@10 |
|---------------|----------|-----------|-----|---------|
| Semantic | 0.65 | 0.72 | 0.68 | 0.71 |
| Semantic + Rerank | 0.68 | 0.75 | 0.70 | 0.73 |
| Keyword (BM25) | 0.58 | 0.68 | 0.62 | 0.64 |
| Keyword + Rerank | 0.61 | 0.71 | 0.64 | 0.67 |
| Hybrid (RRF) | 0.75 | 0.82 | 0.77 | 0.79 |
| **Hybrid + Rerank** | **0.78** | **0.85** | **0.80** | **0.82** |

**Key Finding**: Hybrid retrieval + reranking achieves **+20% Recall@5** improvement vs semantic-only baseline.

## 🚀 Running Evaluation

```bash
# Install dependencies
pip install -e ".[dev]"

# Run evaluation
python -m eval.run_eval_enhanced

# Results saved to:
# - eval/report.json (metrics)
# - eval/report.md (formatted report)
```

## 📈 Configuration (v0.2.0 NEW)

### LLM Settings
```env
LLM_PROVIDER=gemini              # gemini or fake
LLM_MODEL=gemini-1.5-flash       # Model name
LLM_TEMPERATURE=0.1              # Generation temperature
LLM_MAX_TOKENS=1024              # Max tokens
GEMINI_API_KEY=your-key-here     # API key
ABSTAIN_THRESHOLD=0.35           # Confidence threshold
```

### Retrieval Settings
```env
VECTOR_STORE_TYPE=chromadb       # chromadb or qdrant
TOP_K=5                          # Default results count
MIN_SIMILARITY_SCORE=0.0         # Filter threshold
```

### Security Settings (NEW)
```env
API_KEYS=key1:tenant1,key2:tenant2  # Format: key:tenant pairs
```

### Observability Settings (NEW)
```env
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317  # Jaeger/Tempo endpoint (optional)
LOG_LEVEL=INFO                   # Logging level
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage report
make test-cov

# Run specific test suite
pytest tests/generation/ -v
pytest tests/retrieval/ -v
pytest tests/api/ -v
pytest tests/security/ -v
pytest tests/observability/ -v
```

## 📚 Documentation

- **[CHANGELOG.md](CHANGELOG.md)** - Version history and new features
- **[docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md)** - Architecture decisions explained
- **[PRODUCTION_READINESS.md](PRODUCTION_READINESS.md)** - Production deployment checklist
- **[Makefile](Makefile)** - Common development tasks

## 🔧 Development

```bash
# Format code
ruff check rag_pipeline tests --fix

# Type checking
mypy rag_pipeline --ignore-missing-imports

# Run linter
ruff check rag_pipeline tests

# Run tests
pytest tests/ -q

# Start development server with auto-reload
uvicorn rag_pipeline.api:app --reload
```

## 📦 Dependencies

Core dependencies:
- **FastAPI** - Web framework
- **sentence-transformers** - Embeddings and reranking
- **chromadb** / **qdrant-client** - Vector stores
- **google-genai** - Gemini LLM client
- **structlog** - Structured logging
- **opentelemetry-sdk** - Tracing (optional)
- **rank-bm25** - Keyword search

See `pyproject.toml` for complete list.

## 🛣️ Roadmap

### Completed (v0.2.0)
✅ Hybrid retrieval with RRF fusion  
✅ Cross-encoder reranking  
✅ LLM-powered answer generation  
✅ API key authentication  
✅ Multi-tenant support  
✅ Structured logging + tracing  
✅ Async ingestion  
✅ Comprehensive evaluation framework  

### Planned (Future)
- Redis caching layer
- GraphQL API
- Fine-tuned reranker on domain data
- Streaming LLM responses
- Multi-region deployment
- GDPR/CCPA compliance features
- Advanced analytics dashboard

## 📄 License

[Your License Here]

## 👥 Contributing

[Contribution guidelines here]

## 📞 Support

For issues, questions, or feedback:
- Open a GitHub issue
- Check existing documentation
- Review design decisions in docs/DESIGN_DECISIONS.md

---

**Last Updated**: October 2026
**Version**: 0.2.0 (Production-Ready)
