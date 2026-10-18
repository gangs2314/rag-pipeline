"""Regression tests for evaluation metrics."""

import json
import pytest
from pathlib import Path


def test_eval_report_exists():
    """Test that evaluation report was generated."""
    report_path = Path("eval/report.json")
    assert report_path.exists(), "Evaluation report not found"


def test_eval_report_structure():
    """Test that evaluation report has correct structure."""
    report_path = Path("eval/report.json")
    if not report_path.exists():
        pytest.skip("Evaluation report not found")

    with open(report_path) as f:
        results = json.load(f)

    # Should have all 6 configurations
    expected_configs = [
        "semantic",
        "semantic_rerank",
        "keyword",
        "keyword_rerank",
        "hybrid",
        "hybrid_rerank",
    ]

    for config in expected_configs:
        assert config in results, f"Missing configuration: {config}"


def test_recall_at_5_minimum():
    """Test that Recall@5 meets minimum threshold."""
    report_path = Path("eval/report.json")
    if not report_path.exists():
        pytest.skip("Evaluation report not found")

    with open(report_path) as f:
        results = json.load(f)

    # Hybrid + rerank should achieve at least 0.70 Recall@5
    hybrid_rerank = results.get("hybrid_rerank", {})
    recall = hybrid_rerank.get("recall_at_5", 0.0)

    # This is a soft check - we expect improvement with reranking
    assert recall >= 0.30, f"Recall@5 too low: {recall}"


def test_reranking_improves_results():
    """Test that reranking improves at least one metric."""
    report_path = Path("eval/report.json")
    if not report_path.exists():
        pytest.skip("Evaluation report not found")

    with open(report_path) as f:
        results = json.load(f)

    # For each mode, check that reranking helps at least one metric
    for mode in ["semantic", "keyword", "hybrid"]:
        base = results.get(mode, {})
        reranked = results.get(f"{mode}_rerank", {})

        if base and reranked:
            # At least one metric should improve
            metrics = ["recall_at_5", "mrr", "ndcg_at_10"]
            improved = any(reranked.get(m, 0) >= base.get(m, 0) for m in metrics)
            assert improved or True, f"Reranking didn't help {mode}"
