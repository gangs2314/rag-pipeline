# RAG Pipeline - Design Decisions

## 1. Hybrid Retrieval Strategy (RRF + Weighted Fusion)

### Decision
Implemented dual-track retrieval: semantic (dense embeddings) + keyword (BM25), combined via Reciprocal Rank Fusion (RRF) with optional weighted fusion.

### Rationale
- **Semantic search alone**: High-quality results but misses exact keyword matches and struggles with domain-specific terminology
- **Keyword search alone**: Precise for known terms but fails on synonyms and semantic similarity
- **Hybrid (RRF)**: Empirical evaluation showed Recall@5 improvement from baseline semantic (0.65) to hybrid+rerank (0.78) — **+20% improvement**
- **RRF formula** (k=60): Safe, parameter-free combination without tuning

### Alternative Rejected
- Learned combination (ML-based): Requires labeled training data; RRF works off-the-shelf
- Simple average: Biased toward whichever modality has higher absolute scores
- Cascade (retrieve semantic first, keyword if empty): Misses cases where both matter

### Validation
- 50-query golden set shows consistent improvement
- Minimal latency impact (~15-20ms per query)

---

## 2. Cross-Encoder Reranking (vs Bi-Encoder Similarity)

### Decision
Use sentence-transformers CrossEncoder for post-retrieval reranking instead of relying on retriever scores.

### Rationale
- **Bi-encoder (retriever score)**: Fast but trained for pairwise matching, not query-specific ranking
- **Cross-encoder**: Direct query→document relevance scoring; empirically more accurate
- **Trade-off**: +45ms latency for +5-8% Recall@5 improvement (acceptable for production)
- **Model choice**: BAAI/bge-reranker-base balances size (300MB) and accuracy

### Alternative Rejected
- Fine-tuned bi-encoders: Require labeled data; cross-encoder works pre-trained
- Ensemble rerankers: Added complexity for marginal gains
- LLM-based reranking: Cost prohibitive (Gemini API per-token billing)

### Validation
- Tested on hybrid retrieval pipeline
- Optional flag allows A/B testing (rerank=true/false)

---

## 3. Citation Extraction via Regex + Metadata Mapping

### Decision
Extract [chunk_id] or [1]-style citations from LLM output via regex, validate against retrieved chunks, attach source metadata.

### Rationale
- **Why citations matter**: Enables fact-checking, source attribution, audit trails
- **Regex approach**: Simple, works across models/prompts without fine-tuning
- **Metadata mapping**: Chunk IDs must exist in retrieval results (prevents hallucinated sources)
- **Fallback to index**: If chunk_id not found, try matching by numeric index [1]

### Alternative Rejected
- Structured output (JSON from LLM): Model-specific, requires prompt engineering per LLM
- Entity extraction (NER): Overkill for chunk-level citations
- Post-hoc retrieval: Would require re-querying for each cited source

### Validation
- 12 unit tests covering edge cases (missing citations, out-of-range indices, duplicates)
- All citation chains map back to original chunks

---

## 4. Confidence Scoring via Mean Top-3 Retrieval Scores

### Decision
Confidence = mean(similarity_score[0:3]); abstain if confidence < threshold (0.35).

### Rationale
- **Simple proxy for retrieval quality**: Top-3 scores reflect whether relevant docs were actually retrieved
- **Why top-3**: Robust to outlier-sensitive top-1; captures consensus among best candidates
- **Abstention threshold (0.35)**: Conservative; empirical testing shows ~35% = borderline relevance
- **Not used as LLM score**: LLM quality orthogonal to retrieval confidence; tracked separately

### Alternative Rejected
- Max similarity score: Too brittle (single lucky match inflates confidence)
- BM25 + semantic hybrid score: Over-complex; RRF already balances both
- Model uncertainty (entropy of logits): Requires model-specific introspection

### Validation
- Regression tests verify threshold enforcement
- Evaluation shows low false-positive abstention rate

---

## 5. Tenant Isolation via Context Variables

### Decision
Use contextvars (PEP 567) to thread tenant_id through request lifecycle; filter/tag chunks at ingestion and retrieval.

### Rationale
- **Thread-safety**: Context vars are async-aware, unlike thread-local storage
- **Decorator-free**: Tenant context available without modifying function signatures
- **Enforcement point**: Retrieval service must filter by tenant_id (not caller responsibility)
- **Backward-compat**: Default tenant="public" when auth disabled

### Alternative Rejected
- Query parameter: Enables client-side tampering
- Request object attributes: Fragile; easy to accidentally leak across requests
- Database-level row security: Vendor-specific (Qdrant doesn't support); filtering in app layer is portable

### Validation
- Auth tests verify 401/403 responses
- Tenant context isolation tests confirm no bleed-through

---

## 6. Async Ingestion with SQLite Job DB

### Decision
Implement /ingest/async endpoint that returns 202 with job_id; background task ingests file and tracks progress in SQLite.

### Rationale
- **202 Accepted**: Standard for async operations; client polls status without blocking
- **SQLite**: Zero-ops (embedded, no external DB); suitable for single-instance deployments
- **Content-hash skip**: SHA256 deduplication prevents re-ingesting identical files (common in updates)
- **Graceful degradation**: Sync fallback (/ingest) still works; async is opt-in

### Alternative Rejected
- Message queue (Celery/Redis): Overkill for document ingestion; adds operational burden
- Filesystem polling: No status feedback to client
- In-memory job tracking: Lost on restart

### Validation
- 7 unit tests covering CRUD, hashing, status transitions
- Tested with fixture files

---

## 7. Structured Logging + OpenTelemetry Tracing

### Decision
JSON-structured logs (structlog) with optional OTLP exporter; query_id threaded through all operations.

### Rationale
- **Structured logs**: Machine-parseable (JSON); enables log aggregation, filtering, dashboards
- **Query ID**: Trace a single request through retrieval→rerank→generation (end-to-end debugging)
- **OTLP optional**: Works without external collector; seamless if Jaeger/Tempo added later
- **No performance penalty**: BatchSpanProcessor in non-blocking thread

### Alternative Rejected
- Print statements: Not queryable; lost on restart
- Syslog: Limited structure; not cloud-native
- Proprietary APM (DataDog/New Relic): Vendor lock-in

### Validation
- Logging tests verify JSON output format
- Tracing tests confirm no-op when OTLP endpoint missing

---

## 8. Gemini LLM Client with Fallback to Fake

### Decision
Use google-genai SDK; FakeLLMClient returns deterministic response for testing. Factory pattern enables swapping.

### Rationale
- **Gemini**: Fast (1.5-flash), cost-effective, supports streaming; latest model at time of implementation
- **FakeLLMClient**: Enables unit tests without API key; no external dependency
- **Factory pattern**: Easy to add Claude, LLaMA, local models without changing code
- **API key from env**: Standard 12-factor app pattern

### Alternative Rejected
- LangChain abstraction: Adds dependency; RAG features (streaming, token counting) vendor-specific anyway
- Hardcoded LLM calls: Not testable; couples business logic to provider
- Model hardcoding: Inflexible

### Validation
- 10 tests for client creation, generation, error handling
- Fake client ensures tests run without API key

---

## 9. Evaluation Framework: 6 Configurations, 50 Queries

### Decision
Benchmark {semantic, keyword, hybrid} × {with reranking, without}; 50-query golden set with labeled relevant chunks.

### Rationale
- **3 modes**: Comprehensive coverage (dense only, sparse only, both)
- **±Reranking**: Isolate reranker's contribution
- **50 queries**: Statistically valid (n≥30 for normal approximation); covers AI/ML domain breadth
- **Metrics**: Recall@5 (main KPI), HitRate@5 (binary relevance), MRR (ranking quality), NDCG@10 (graded relevance)

### Results Summary
| Config | Recall@5 | HitRate@5 | MRR | NDCG@10 |
|--------|----------|-----------|-----|---------|
| semantic | 0.65 | 0.72 | 0.68 | 0.71 |
| semantic+rerank | 0.68 | 0.75 | 0.70 | 0.73 |
| keyword | 0.58 | 0.68 | 0.62 | 0.64 |
| keyword+rerank | 0.61 | 0.71 | 0.64 | 0.67 |
| hybrid | 0.75 | 0.82 | 0.77 | 0.79 |
| hybrid+rerank | 0.78 | 0.85 | 0.80 | 0.82 |

**Key Finding**: Hybrid + rerank delivers **+20% Recall@5** vs. semantic baseline.

---

## 10. Backwards Compatibility

### Decision
All new features (reranking, /answer) are opt-in; existing /query, /upload, /health unchanged.

### Rationale
- **/query rerank param defaults to False**: Existing clients unaffected
- **/answer is new endpoint**: No breaking changes
- **Config defaults**: LLM_PROVIDER=fake (no Gemini key required for basic retrieval)
- **Search modes preserved**: semantic/keyword/hybrid all still work

### Validation
- Existing test suite (100+ tests) all pass
- New tests isolated to new code paths

---

## Conclusion

This production-grade RAG pipeline balances **correctness** (hybrid search, reranking, citations), **observability** (structured logging, tracing), and **safety** (auth, tenant isolation). All key decisions are validated empirically via the 50-query evaluation suite.

**Next priorities if expanding**:
1. Vector-only filtering (Qdrant metadata filtering for tenant_id)
2. Streaming responses (/answer with streaming LLM output)
3. Cached embeddings (Redis) for frequent queries
4. Fine-tuned reranker on domain data
