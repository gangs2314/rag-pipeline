"""Tests for observability layer."""



from rag_pipeline.observability.logging import RequestLogger, get_logger


def test_get_logger():
    """Test logger creation."""
    logger = get_logger(__name__)
    assert logger is not None


def test_request_logger_creation():
    """Test RequestLogger initialization."""
    logger = RequestLogger()
    assert logger is not None


def test_request_logger_methods_exist():
    """Test that RequestLogger has all required methods."""
    logger = RequestLogger()

    assert hasattr(logger, "log_request")
    assert hasattr(logger, "log_retrieval")
    assert hasattr(logger, "log_reranking")
    assert hasattr(logger, "log_generation")
    assert hasattr(logger, "log_error")


def test_log_request():
    """Test logging a request."""
    logger = RequestLogger()

    # Should not raise
    logger.log_request(
        query_id="test-123",
        endpoint="/query",
        query="test query",
        mode="semantic",
        rerank=False,
    )


def test_log_retrieval():
    """Test logging retrieval operation."""
    logger = RequestLogger()

    logger.log_retrieval(
        query_id="test-123",
        mode="semantic",
        query_text="machine learning",
        num_results=5,
        latency_ms=125.5,
    )


def test_log_reranking():
    """Test logging reranking operation."""
    logger = RequestLogger()

    logger.log_reranking(
        query_id="test-123",
        num_candidates=10,
        latency_ms=45.2,
    )


def test_log_generation():
    """Test logging generation operation."""
    logger = RequestLogger()

    logger.log_generation(
        query_id="test-123",
        confidence=0.85,
        abstained=False,
        latency_ms=320.1,
        tokens_in=150,
        tokens_out=75,
    )


def test_log_error():
    """Test logging error."""
    logger = RequestLogger()

    logger.log_error(
        query_id="test-123",
        operation="retrieval",
        error="Query failed",
        error_type="ValueError",
    )


def test_tracing_init():
    """Test tracing initialization."""
    from rag_pipeline.observability.tracing import init_tracing

    # Should not raise even without OTLP endpoint
    tracer = init_tracing()
    assert tracer is not None


def test_get_tracer():
    """Test tracer retrieval."""
    from rag_pipeline.observability.tracing import get_tracer

    tracer = get_tracer(__name__)
    assert tracer is not None


def test_rag_spans():
    """Test RAGSpans creation."""
    from rag_pipeline.observability.tracing import RAGSpans

    spans = RAGSpans()
    assert spans is not None
