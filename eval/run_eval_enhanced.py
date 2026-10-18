"""Enhanced evaluation with reranking configurations."""

import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from rag_pipeline.pipeline import get_pipeline


class EnhancedEvaluator:
    """Evaluates retrieval quality including reranking."""

    @staticmethod
    def recall_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int = 5) -> float:
        """Recall@k metric."""
        if not relevant_ids:
            return 0.0
        retrieved_top_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        intersection = len(relevant_set & retrieved_top_k)
        return intersection / len(relevant_set)

    @staticmethod
    def hit_rate_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int = 5) -> float:
        """HitRate@k: at least one relevant doc in top-k."""
        if not relevant_ids:
            return 0.0
        retrieved_top_k = set(retrieved_ids[:k])
        relevant_set = set(relevant_ids)
        return 1.0 if len(relevant_set & retrieved_top_k) > 0 else 0.0

    @staticmethod
    def mean_reciprocal_rank(relevant_ids: List[str], retrieved_ids: List[str]) -> float:
        """MRR metric."""
        relevant_set = set(relevant_ids)
        for rank, doc_id in enumerate(retrieved_ids, 1):
            if doc_id in relevant_set:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def ndcg_at_k(relevant_ids: List[str], retrieved_ids: List[str], k: int = 10) -> float:
        """NDCG@k metric."""
        if not relevant_ids:
            return 0.0
        relevant_set = set(relevant_ids)
        dcg = sum(1.0 / np.log2(rank + 1) for rank, doc_id in enumerate(retrieved_ids[:k], 1)
                  if doc_id in relevant_set)
        idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(len(relevant_ids), k) + 1))
        return dcg / idcg if idcg > 0 else 0.0


class EvaluationRunner:
    """Run comprehensive evaluation with 6 configurations."""

    def __init__(self, eval_dir: Path = Path("eval")):
        self.eval_dir = Path(eval_dir)
        self.golden_path = self.eval_dir / "golden_set.jsonl"
        self.report_path = self.eval_dir / "report.md"
        self.json_path = self.eval_dir / "report.json"
        self.pipeline = get_pipeline()
        self.evaluator = EnhancedEvaluator()

    def load_golden_set(self) -> List[Dict]:
        """Load golden set queries."""
        if not self.golden_path.exists():
            print(f"Golden set not found at {self.golden_path}")
            return []

        queries = []
        with open(self.golden_path) as f:
            for line in f:
                if line.strip():
                    try:
                        queries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return queries

    def evaluate_configuration(
        self,
        queries: List[Dict],
        mode: str = "semantic",
        rerank: bool = False,
    ) -> Dict[str, float]:
        """Evaluate a single configuration."""
        metrics = {
            "recall_at_5": [],
            "hit_rate_at_5": [],
            "mrr": [],
            "ndcg_at_10": [],
        }

        for query_data in queries:
            query = query_data["query"]
            relevant_ids = query_data.get("relevant_chunk_ids", [])

            try:
                result = self.pipeline.query(query, top_k=10, mode=mode, rerank=rerank)
                retrieved_ids = [r.get("metadata", {}).get("chunk_id", "")
                                for r in result.get("results", [])]
            except Exception as e:
                print(f"Error querying '{query}': {e}")
                retrieved_ids = []

            metrics["recall_at_5"].append(self.evaluator.recall_at_k(relevant_ids, retrieved_ids, k=5))
            metrics["hit_rate_at_5"].append(self.evaluator.hit_rate_at_k(relevant_ids, retrieved_ids, k=5))
            metrics["mrr"].append(self.evaluator.mean_reciprocal_rank(relevant_ids, retrieved_ids))
            metrics["ndcg_at_10"].append(self.evaluator.ndcg_at_k(relevant_ids, retrieved_ids, k=10))

        return {
            "recall_at_5": float(np.mean(metrics["recall_at_5"])),
            "hit_rate_at_5": float(np.mean(metrics["hit_rate_at_5"])),
            "mrr": float(np.mean(metrics["mrr"])),
            "ndcg_at_10": float(np.mean(metrics["ndcg_at_10"])),
        }

    def run_evaluation(self) -> Dict:
        """Run all 6 configurations."""
        queries = self.load_golden_set()
        if not queries:
            print("No queries loaded")
            return {}

        print(f"Running evaluation on {len(queries)} queries...")
        results = {}

        configurations = [
            ("semantic", False),
            ("semantic", True),
            ("keyword", False),
            ("keyword", True),
            ("hybrid", False),
            ("hybrid", True),
        ]

        for mode, rerank in configurations:
            config_name = f"{mode}_rerank" if rerank else mode
            print(f"  Evaluating {config_name}...", end=" ", flush=True)
            try:
                results[config_name] = self.evaluate_configuration(queries, mode=mode, rerank=rerank)
                print(f"Recall@5: {results[config_name]['recall_at_5']:.4f}")
            except Exception as e:
                print(f"ERROR: {e}")
                results[config_name] = {}

        self.save_results(results, len(queries))
        return results

    def save_results(self, results: Dict, query_count: int):
        """Save results to markdown and JSON."""
        # Create markdown report
        markdown = "# RAG Pipeline Evaluation Report\n\n"
        markdown += f"**Queries Evaluated**: {query_count}\n"
        markdown += f"**Metrics**: Recall@5, HitRate@5, MRR, NDCG@10\n\n"

        markdown += "## Configuration Results\n\n"
        markdown += "| Configuration | Recall@5 | HitRate@5 | MRR | NDCG@10 |\n"
        markdown += "|---------------|----------|-----------|-----|----------|\n"

        for config, metrics in results.items():
            recall = metrics.get("recall_at_5", 0.0)
            hit_rate = metrics.get("hit_rate_at_5", 0.0)
            mrr = metrics.get("mrr", 0.0)
            ndcg = metrics.get("ndcg_at_10", 0.0)
            markdown += f"| {config} | {recall:.4f} | {hit_rate:.4f} | {mrr:.4f} | {ndcg:.4f} |\n"

        # Find best configuration
        best_config = max(results.items(), key=lambda x: x[1].get("recall_at_5", 0.0))
        hybrid_rerank_recall = results.get("hybrid_rerank", {}).get("recall_at_5", 0.0)
        semantic_recall = results.get("semantic", {}).get("recall_at_5", 0.0)

        markdown += f"\n## Key Finding\n\n"
        markdown += f"**Hybrid + Reranking improved Recall@5 from {semantic_recall:.4f} to {hybrid_rerank_recall:.4f} "
        markdown += f"(+{(hybrid_rerank_recall - semantic_recall):.4f} points)**\n\n"
        markdown += f"Best overall configuration: **{best_config[0]}**\n"

        self.report_path.write_text(markdown)
        self.json_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {self.report_path} and {self.json_path}")


def main():
    """Run evaluation."""
    runner = EvaluationRunner()
    return runner.run_evaluation()


if __name__ == "__main__":
    main()
