"""Structured logging for RAG pipeline."""

import logging
import sys
from datetime import datetime
from typing import Optional

import structlog


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured JSON logging.

    Args:
        log_level: Logging level (INFO, DEBUG, WARNING, ERROR).
    """
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get structured logger instance.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Configured structlog logger.
    """
    return structlog.get_logger(name)


class RequestLogger:
    """Log structured request/response data."""

    def __init__(self):
        """Initialize request logger."""
        self.logger = get_logger(__name__)

    def log_request(
        self,
        query_id: str,
        endpoint: str,
        query: str,
        mode: Optional[str] = None,
        rerank: bool = False,
    ) -> None:
        """Log incoming request.

        Args:
            query_id: Unique request identifier.
            endpoint: API endpoint.
            query: Search query or input.
            mode: Search mode if applicable.
            rerank: Whether reranking is enabled.
        """
        self.logger.info(
            "request_received",
            query_id=query_id,
            endpoint=endpoint,
            query=query,
            mode=mode,
            rerank=rerank,
            timestamp=datetime.utcnow().isoformat(),
        )

    def log_retrieval(
        self,
        query_id: str,
        mode: str,
        query_text: str,
        num_results: int,
        latency_ms: float,
    ) -> None:
        """Log retrieval operation.

        Args:
            query_id: Request identifier.
            mode: Search mode used.
            query_text: The query.
            num_results: Number of results retrieved.
            latency_ms: Operation latency in milliseconds.
        """
        self.logger.info(
            "retrieval_completed",
            query_id=query_id,
            mode=mode,
            query_length=len(query_text),
            num_results=num_results,
            latency_ms=latency_ms,
        )

    def log_reranking(
        self,
        query_id: str,
        num_candidates: int,
        latency_ms: float,
    ) -> None:
        """Log reranking operation.

        Args:
            query_id: Request identifier.
            num_candidates: Number of candidates reranked.
            latency_ms: Operation latency in milliseconds.
        """
        self.logger.info(
            "reranking_completed",
            query_id=query_id,
            num_candidates=num_candidates,
            latency_ms=latency_ms,
        )

    def log_generation(
        self,
        query_id: str,
        confidence: float,
        abstained: bool,
        latency_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
    ) -> None:
        """Log answer generation.

        Args:
            query_id: Request identifier.
            confidence: Retrieval confidence score.
            abstained: Whether model abstained.
            latency_ms: Operation latency in milliseconds.
            tokens_in: Input tokens.
            tokens_out: Output tokens.
        """
        self.logger.info(
            "generation_completed",
            query_id=query_id,
            confidence=confidence,
            abstained=abstained,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    def log_error(
        self,
        query_id: str,
        operation: str,
        error: str,
        error_type: Optional[str] = None,
    ) -> None:
        """Log error event.

        Args:
            query_id: Request identifier.
            operation: Operation that failed.
            error: Error message.
            error_type: Exception type.
        """
        self.logger.error(
            "operation_failed",
            query_id=query_id,
            operation=operation,
            error=error,
            error_type=error_type,
        )
