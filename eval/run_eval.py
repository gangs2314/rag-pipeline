"""Evaluation harness for RAG pipeline retrieval quality."""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from rag_pipeline.pipeline import get_pipeline


class RetrievalEvaluator:
    """Evaluates retrieval quality using standard metrics."""

    @staticmethod
    def recall_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int = 5) -> float:
        """
        Compute Recall@k: fraction of relevant docs in top-k results.

        Recall@k = |relevant ∩ retrieved_top_k| / |relevant|
        """
        if not relevant_ids:
            return 0.0

        retrieved_top_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        intersection = len(relevant_set & retrieved_top_k)

        return intersection / len(relevant_set)

    @staticmethod
    def mean_reciprocal_rank(relevant_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Compute MRR: reciprocal rank of first relevant document.

        MRR = 1 / rank of first relevant document
        Returns 0.0 if no relevant document in results
        """
        relevant_set = set(relevant_ids)

        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                return 1.0 / rank

        return 0.0

    @staticmethod
    def ndcg_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int = 10) -> float:
        """
        Compute NDCG@k: Normalized Discounted Cumulative Gain.

        DCG@k = Σ(1 / log2(rank+1)) for relevant documents in top-k
        NDCG@k = DCG@k / IDCG@k (ideal DCG)
        """
        if not relevant_ids:
            return 0.0

        # Compute DCG
        relevant_set = set(relevant_ids)
        dcg = 0.0

        for rank, doc_id in enumerate(retrieved_ids[:k], 1):
            if doc_id in relevant_set:
                dcg += 1.0 / np.log2(rank + 1)

        # Compute IDCG (ideal: all relevant docs ranked first)
        idcg = 0.0
        for rank in range(1, min(len(relevant_ids), k) + 1):
            idcg += 1.0 / np.log2(rank + 1)

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def precision_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int = 5) -> float:
        """
        Compute Precision@k: fraction of top-k results that are relevant.

        Precision@k = |relevant ∩ retrieved_top_k| / k
        """
        if k == 0:
            return 0.0

        retrieved_top_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        intersection = len(relevant_set & retrieved_top_k)

        return intersection / min(k, len(retrieved_ids))


class EvaluationRunner:
    """Runs evaluation on the RAG pipeline."""

    def __init__(self, eval_dir: Path = Path("eval")):
        """Initialize evaluator."""
        self.eval_dir = Path(eval_dir)
        self.documents_path = self.eval_dir / "documents.json"
        self.queries_path = self.eval_dir / "queries.json"
        self.results_path = self.eval_dir / "results.md"
        self.pipeline = get_pipeline()
        self.evaluator = RetrievalEvaluator()
        # Build filename to doc_id mapping
        self.filename_to_doc_id = {}

    def load_eval_data(self) -> Tuple[List[Dict], List[Dict]]:
        """Load evaluation data."""
        with open(self.documents_path) as f:
            documents = json.load(f)

        with open(self.queries_path) as f:
            queries = json.load(f)

        return documents, queries

    def ingest_documents(self, documents: List[Dict]):
        """Ingest documents into pipeline."""
        print(f"Ingesting {len(documents)} documents...")

        # Create temporary files for each document
        import tempfile
        temp_dir = Path(tempfile.gettempdir()) / "rag_eval"
        temp_dir.mkdir(exist_ok=True)

        for doc in documents:
            doc_path = temp_dir / doc["filename"]
            doc_path.write_text(doc["content"])

            # Build filename -> doc_id mapping
            self.filename_to_doc_id[doc["filename"]] = doc["id"]

            result = self.pipeline.ingest_document(doc_path)
            status = "[OK]" if result["status"] == "success" else "[FAIL]"
            print(f"  {status} {doc['filename']}: {result.get('chunk_count', 0)} chunks")

    def evaluate_mode(
        self,
        queries: List[Dict],
        mode: str = "semantic"
    ) -> Dict[str, float]:
        """Evaluate a single search mode."""
        print(f"\nEvaluating {mode.upper()} search mode...")

        metrics = {
            "recall_at_5": [],
            "precision_at_5": [],
            "mrr": [],
            "ndcg_at_10": [],
        }

        for query_data in queries:
            query = query_data["query"]
            relevant_docs = query_data["relevant_docs"]

            # Query pipeline
            try:
                result = self.pipeline.query(query, top_k=10, mode=mode)

                # Extract document IDs from results using filename mapping
                retrieved_ids = []
                for r in result["results"]:
                    source_file = r["metadata"].get("source_file", "")
                    if source_file in self.filename_to_doc_id:
                        doc_id = self.filename_to_doc_id[source_file]
                        retrieved_ids.append(doc_id)

            except Exception as e:
                print(f"  Error querying '{query}': {e}")
                retrieved_ids = []

            # Compute metrics
            recall_5 = self.evaluator.recall_at_k(relevant_docs, retrieved_ids, k=5)
            precision_5 = self.evaluator.precision_at_k(relevant_docs, retrieved_ids, k=5)
            mrr = self.evaluator.mean_reciprocal_rank(relevant_docs, retrieved_ids)
            ndcg_10 = self.evaluator.ndcg_at_k(relevant_docs, retrieved_ids, k=10)

            metrics["recall_at_5"].append(recall_5)
            metrics["precision_at_5"].append(precision_5)
            metrics["mrr"].append(mrr)
            metrics["ndcg_at_10"].append(ndcg_10)

        # Compute averages
        avg_metrics = {
            "recall_at_5": float(np.mean(metrics["recall_at_5"])),
            "precision_at_5": float(np.mean(metrics["precision_at_5"])),
            "mrr": float(np.mean(metrics["mrr"])),
            "ndcg_at_10": float(np.mean(metrics["ndcg_at_10"])),
        }

        print(f"  Recall@5: {avg_metrics['recall_at_5']:.4f}")
        print(f"  Precision@5: {avg_metrics['precision_at_5']:.4f}")
        print(f"  MRR: {avg_metrics['mrr']:.4f}")
        print(f"  NDCG@10: {avg_metrics['ndcg_at_10']:.4f}")

        return avg_metrics

    def run_evaluation(self) -> Dict[str, Dict]:
        """Run full evaluation across all modes."""
        print("=" * 60)
        print("RAG Pipeline Evaluation Harness")
        print("=" * 60)

        # Load data
        documents, queries = self.load_eval_data()

        # Ingest documents
        self.ingest_documents(documents)

        # Evaluate each mode
        results = {}
        modes = ["semantic", "keyword", "hybrid"]

        for mode in modes:
            try:
                results[mode] = self.evaluate_mode(queries, mode=mode)
            except Exception as e:
                print(f"Error evaluating {mode} mode: {e}")
                results[mode] = {}

        # Save results
        self.save_results(results, queries)

        return results

    def save_results(self, results: Dict[str, Dict], queries: List[Dict]):
        """Save evaluation results to markdown file."""
        markdown = "# RAG Pipeline Evaluation Results\n\n"
        markdown += f"**Date**: {self.get_timestamp()}\n\n"
        markdown += f"**Total Queries**: {len(queries)}\n"
        markdown += f"**Evaluation Metrics**: Recall@5, Precision@5, MRR, NDCG@10\n\n"

        markdown += "## Summary Results\n\n"
        markdown += "| Metric | Semantic | Keyword | Hybrid |\n"
        markdown += "|--------|----------|---------|--------|\n"

        metrics_to_show = ["recall_at_5", "precision_at_5", "mrr", "ndcg_at_10"]
        metric_names = {
            "recall_at_5": "Recall@5",
            "precision_at_5": "Precision@5",
            "mrr": "MRR",
            "ndcg_at_10": "NDCG@10",
        }

        for metric in metrics_to_show:
            row = f"| {metric_names[metric]} |"
            for mode in ["semantic", "keyword", "hybrid"]:
                value = results.get(mode, {}).get(metric, 0.0)
                row += f" {value:.4f} |"
            markdown += row + "\n"

        markdown += "\n## Observations\n\n"
        markdown += "### Semantic Search (Dense Vector Similarity)\n"
        markdown += "- Uses sentence embeddings for semantic understanding\n"
        markdown += "- Best for finding conceptually similar documents\n"
        markdown += f"- Recall@5: {results.get('semantic', {}).get('recall_at_5', 0.0):.4f}\n\n"

        markdown += "### Keyword Search (BM25)\n"
        markdown += "- Uses statistical term frequency-inverse document frequency\n"
        markdown += "- Best for exact term matching\n"
        markdown += f"- Recall@5: {results.get('keyword', {}).get('recall_at_5', 0.0):.4f}\n\n"

        markdown += "### Hybrid Search (Combined with Reranking)\n"
        markdown += "- Combines semantic + keyword with reciprocal rank fusion\n"
        markdown += "- Cross-encoder reranking improves result ordering\n"
        markdown += f"- Recall@5: {results.get('hybrid', {}).get('recall_at_5', 0.0):.4f}\n\n"

        markdown += "## Conclusion\n\n"

        # Find best mode
        best_mode = max(
            results.keys(),
            key=lambda m: results[m].get('recall_at_5', 0.0)
        )

        markdown += f"Best performing mode: **{best_mode.upper()}**\n\n"
        markdown += "Hybrid search combines the strengths of semantic and keyword search,\n"
        markdown += "providing balanced performance across different query types.\n"

        # Save to file
        self.results_path.write_text(markdown, encoding='utf-8')
        print(f"\n[OK] Results saved to {self.results_path}")


    @staticmethod
    def get_timestamp() -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()


def main():
    """Run evaluation."""
    runner = EvaluationRunner()
    results = runner.run_evaluation()
    return results


if __name__ == "__main__":
    main()
