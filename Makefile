.PHONY: help install lint test test-cov eval run clean docs

help:
	@echo "RAG Pipeline - Production-Grade Retrieval-Augmented Generation"
	@echo ""
	@echo "Available targets:"
	@echo "  install       Install dependencies"
	@echo "  lint          Run ruff linter"
	@echo "  type-check    Run mypy type checker"
	@echo "  test          Run tests"
	@echo "  test-cov      Run tests with coverage report"
	@echo "  eval          Run evaluation suite"
	@echo "  run           Start API server"
	@echo "  docs          Generate documentation"
	@echo "  clean         Clean up artifacts"

install:
	pip install -e ".[dev]"

lint:
	ruff check rag_pipeline tests eval
	@echo "✓ Linting passed"

type-check:
	mypy rag_pipeline --ignore-missing-imports
	@echo "✓ Type checking passed"

test:
	pytest tests/ -q --tb=short

test-cov:
	pytest tests/ --cov=rag_pipeline --cov-report=term-missing --cov-report=html
	@echo "✓ Coverage report generated in htmlcov/index.html"

eval:
	python -m eval.run_eval_enhanced
	@echo "✓ Evaluation complete - results in eval/report.json"

run:
	python -m rag_pipeline serve --port 8000

docs:
	@echo "Generating documentation..."
	@echo "# RAG Pipeline Documentation" > docs/README.md
	@echo "" >> docs/README.md
	@echo "See docs/DESIGN_DECISIONS.md for architecture details" >> docs/README.md

clean:
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .coverage htmlcov
	rm -rf __pycache__ **/__pycache__
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	@echo "✓ Cleaned up artifacts"
