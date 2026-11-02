# RAG Pipeline

A working prototype for document ingestion, semantic search, and answer generation. Upload any document (PDF, code, Markdown, JSON, DOCX, HTML, TXT), retrieve relevant chunks, and generate answers with source citations.

## Features

### Document Processing
- Multi-format support: PDF, TXT, Markdown, JSON, code files (Python, JavaScript, TypeScript, Java, C++, Go, Rust), DOCX, HTML
- Chunking strategies:
  - Recursive character splitting (default)
  - Header-aware Markdown splitting
  - Code-aware splitting with language-specific separators
  - Parent-child hierarchical chunking (small chunks for search, large for LLM context)
  - Semantic chunking based on sentence similarity

### Search & Retrieval
- Hybrid retrieval: Combine semantic (dense embeddings) + keyword (BM25) search
- Reciprocal rank fusion (RRF) for combining results
- Optional cross-encoder reranking (BAAI/bge-reranker-base)
- Multiple search modes: semantic, keyword, hybrid

### LLM & Answer Generation
- LLM integration: Gemini API client
- FakeLLMClient for testing without API keys
- Extensible LLM abstraction (swap Claude, LLaMA, etc.)
- Answer generation endpoint with source citations
- Confidence scoring and abstention when confidence is low

### Enterprise Features
- API key authentication (X-API-Key header)
- Multi-tenant isolation via context variables
- Structured logging (JSON with structlog)
- Optional OpenTelemetry tracing
- Async document ingestion with job tracking
- Content-hash deduplication (SHA256)

### Embeddings & Storage
- Sentence-Transformers embeddings (all-MiniLM-L6-v2 by default)
- ChromaDB for local/dev; Qdrant for production
- Multi-strategy deduplication: exact hashing + n-gram similarity (95% threshold)
- Rich metadata per chunk (source, page, hierarchy, hash, etc.)

## Quick Start

### Prerequisites
- Python 3.11+
- pip
- Optional: CUDA-capable GPU

### Installation

```bash
git clone https://github.com/gangs2314/rag-pipeline
cd rag-pipeline
pip install -e .

# For development and testing
pip install -e ".[dev]"
```

### Verify Installation

```bash
python -m rag_pipeline --help
```

### Start the API Server

```bash
python -m rag_pipeline serve --port 8000
```

API available at `http://localhost:8000` with docs at `/docs`.

### Upload a Document

```bash
# CLI
python -m rag_pipeline ingest path/to/document.pdf

# API (curl)
curl -X POST "http://localhost:8000/upload" \
  -F "file=@path/to/document.pdf"
```

### Query Documents

```bash
# CLI
python -m rag_pipeline query "What is the main topic?" --top-k 5

# API (curl)
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "top_k": 5}'
```

## Docker Demo

```bash
docker-compose up
```

This will:
1. Build the container
2. Start the API at http://localhost:8000
3. Load 5 sample documents
4. Launch frontend at http://localhost:80

Access interactive docs at http://localhost:8000/docs

Clean up with:
```bash
docker-compose down -v
```

## Configuration

Create `.env` file to override defaults:

```env
# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Vector Store
VECTOR_STORE_TYPE=chromadb
CHROMADB_PATH=./data/chromadb

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu  # or 'cuda'

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=50
USE_PARENT_CHILD_CHUNKING=true
PARENT_CHUNK_SIZE=1024
CHILD_CHUNK_SIZE=256

# LLM
LLM_PROVIDER=gemini
LLM_MODEL=gemini-1.5-flash
GEMINI_API_KEY=your-key-here

# Retrieval
TOP_K=5
MIN_SIMILARITY_SCORE=0.0
```

## REST API

### Health Check
```
GET /health
```

### Upload Document
```
POST /upload
Content-Type: multipart/form-data
```

Response:
```json
{
  "document_id": "uuid-...",
  "file_name": "document.pdf",
  "chunk_count": 42,
  "status": "success"
}
```

### Query Documents
```
POST /query
Content-Type: application/json
```

Request:
```json
{
  "query": "What is the main topic?",
  "top_k": 5,
  "mode": "semantic"
}
```

Response:
```json
{
  "query": "...",
  "result_count": 5,
  "results": [
    {
      "document_id": "...",
      "content": "...",
      "metadata": {"source_file": "...", "page_number": 1},
      "similarity_score": 0.895
    }
  ],
  "status": "success"
}
```

### Generate Answer
```
POST /answer
Content-Type: application/json
```

Request:
```json
{
  "query": "What is machine learning?",
  "top_k": 5
}
```

Response:
```json
{
  "query": "...",
  "answer": "...",
  "citations": [
    {
      "chunk_id": "...",
      "chunk_index": 1,
      "source_file": "document.pdf",
      "page_number": 3
    }
  ],
  "confidence": 0.92,
  "abstained": false,
  "status": "success"
}
```

### Get Statistics
```
GET /stats
```

Response:
```json
{
  "collection_name": "rag_documents",
  "document_count": 150,
  "vector_store_type": "chromadb",
  "embedding_model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

### Metrics
```
GET /metrics
```

Prometheus-format metrics.

### Delete Document
```
DELETE /documents/{document_id}
```

## CLI Commands

```bash
# Start API server
python -m rag_pipeline serve --port 8000
python -m rag_pipeline serve --reload  # Auto-reload for development

# Ingest document
python -m rag_pipeline ingest path/to/document.pdf

# Query documents
python -m rag_pipeline query "search term" --top-k 5

# View statistics
python -m rag_pipeline stats

# Download Kaggle dataset
python -m rag_pipeline kaggle download manisha717/dataset-of-pdf-files

# List local Kaggle datasets
python -m rag_pipeline kaggle list

# Delete local Kaggle dataset
python -m rag_pipeline kaggle delete manisha717/dataset-of-pdf-files
```

## Supported Document Types

| Format | Extension | Features |
|--------|-----------|----------|
| PDF | `.pdf` | Per-page extraction, page metadata |
| Text | `.txt` | Plain text with encoding detection |
| Markdown | `.md`, `.markdown` | Structure-aware parsing |
| Code | `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.go`, `.rs` | Language-specific metadata |
| JSON | `.json` | Hierarchical flattening |
| DOCX | `.docx` | Word document parsing |
| HTML | `.html` | DOM structure preserved |

## Chunking Strategies

### Recursive Character Splitting (Default)
- Chunk size: 512 characters
- Overlap: 50 characters
- Separators (in priority order): `["\n\n", "\n", ". ", " ", ""]`

### Header-Aware Markdown Splitting
- Splits first by headers (#, ##, ###)
- Then recursively splits within sections
- Preserves header hierarchy in metadata

### Code-Aware Splitting
- Language-specific separators (function/class definitions)
- Preserves code structure and comments
- Supported: Python, JavaScript, TypeScript, Java, C++, Go, Rust

### Parent-Child Hierarchical Chunking
- Child chunks: 256 characters (for similarity search)
- Parent chunks: 1024 characters (returned to LLM for context)
- Child-parent linking via metadata

### Semantic Chunking
- Splits based on sentence similarity
- Similarity threshold: 0.75
- Minimum chunk size: 100 characters

## Embeddings & Vector Store

### Embedding Model
- Default: `sentence-transformers/all-MiniLM-L6-v2` (384 dims)
- Production option: `BAAI/bge-large-en-v1.5` (1024 dims)
- Models cached locally at `./models/embeddings/`

### Vector Store
- Development: ChromaDB (no setup, persistent disk storage)
- Production: Qdrant (distributed, requires running server)

Collection name: `rag_documents`
Distance metric: cosine similarity

## Metadata

Each chunk includes:
```json
{
  "source_file": "document.pdf",
  "doc_type": "pdf",
  "page_number": 1,
  "chunk_index": 0,
  "parent_id": "...",
  "content_hash": "...",
  "chunk_created_at": "2026-10-19T08:00:00Z"
}
```

## Testing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ -v --cov=rag_pipeline

# Linting
ruff check rag_pipeline/ tests/

# Format code
black rag_pipeline/ tests/

# Type checking
mypy rag_pipeline/
```

## Project Structure

```
rag_pipeline/
├── __init__.py
├── __main__.py              # CLI entry point
├── config.py                # Configuration
├── loaders.py               # Document loaders
├── chunking.py              # Chunking strategies
├── deduplication.py         # Deduplication logic
├── vector_store.py          # Vector store abstraction
├── pipeline.py              # Main orchestrator
├── api.py                   # FastAPI endpoints
├── cli.py                   # CLI commands
├── kaggle_integration.py    # Kaggle dataset download
├── generation/              # LLM answer generation
├── retrieval/               # Hybrid search & reranking
├── security/                # Authentication
├── observability/           # Logging & tracing
└── tasks/                   # Async ingestion

data/
├── chromadb/                # Vector store (local)
├── uploads/                 # Document uploads
├── kaggle_datasets/         # Downloaded datasets
└── models/embeddings/       # Cached embeddings

tests/
├── test_chunking.py
├── test_dedup.py
└── ...                      # Other test suites

docs/
├── DESIGN_DECISIONS.md      # Architecture decisions
└── ...

pyproject.toml              # Project config
.env.example                # Environment template
docker-compose.yml          # Docker setup
Dockerfile                  # Container image
nginx.conf                  # Reverse proxy
index.html                  # Web frontend
```

## Known Limitations

TODO: Add limitations discovered during implementation and testing (e.g., performance constraints, unsupported features, known bugs).

## Why I Built This

TODO: 2-3 sentences in your own words about motivation and goals.

## Design Decisions

TODO: Explain key decisions:
- Why ChromaDB for development?
- Why these chunk sizes?
- Why the 95% dedup threshold?
- Why semantic chunking is optional?

## What I Learned

TODO: What did you learn building this? Insights about RAG systems, production considerations, etc.

## Troubleshooting

### PDF loading fails
Install pdfplumber:
```bash
pip install pdfplumber
```

### Embedding model fails to load
- Check internet connection
- Verify disk space (~400MB for all-MiniLM)
- Check model path in `.env`

### ChromaDB errors
- Ensure `./data/chromadb/` is writable
- Check available disk space
- Try removing cache: `rm -rf ./data/chromadb/`

### Out of memory during embedding
- Reduce batch size
- Use CPU instead of GPU: `EMBEDDING_DEVICE=cpu`
- Use smaller model: `EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2`

## License

MIT License - See LICENSE file

## Contributing

Contributions welcome. Please open an issue or pull request.

---

**Last Updated**: October 2026
**Status**: Prototype
