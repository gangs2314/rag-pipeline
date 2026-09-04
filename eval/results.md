# RAG Pipeline Evaluation Results

**Date**: 2026-09-04T13:38:26.359715

**Total Queries**: 15
**Evaluation Metrics**: Recall@5, Precision@5, MRR, NDCG@10

## Summary Results

| Metric | Semantic | Keyword | Hybrid |
|--------|----------|---------|--------|
| Recall@5 | 0.5333 | 0.5333 | 0.5333 |
| Precision@5 | 0.1600 | 0.1600 | 0.1600 |
| MRR | 0.8067 | 0.8067 | 0.8067 |
| NDCG@10 | 2.6974 | 2.6974 | 2.6856 |

## Observations

### Semantic Search (Dense Vector Similarity)
- Uses sentence embeddings for semantic understanding
- Best for finding conceptually similar documents
- Recall@5: 0.5333

### Keyword Search (BM25)
- Uses statistical term frequency-inverse document frequency
- Best for exact term matching
- Recall@5: 0.5333

### Hybrid Search (Combined with Reranking)
- Combines semantic + keyword with reciprocal rank fusion
- Cross-encoder reranking improves result ordering
- Recall@5: 0.5333

## Conclusion

Best performing mode: **SEMANTIC**

Hybrid search combines the strengths of semantic and keyword search,
providing balanced performance across different query types.
