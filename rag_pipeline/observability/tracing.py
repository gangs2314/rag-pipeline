"""OpenTelemetry tracing for RAG pipeline."""

import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def init_tracing(service_name: str = "rag-pipeline") -> TracerProvider:
    """Initialize OpenTelemetry tracing.

    Args:
        service_name: Service name for traces.

    Returns:
        Configured TracerProvider.
    """
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    if not otlp_endpoint:
        # No-op tracer if endpoint not configured
        return TracerProvider()

    try:
        exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(tracer_provider)
        return tracer_provider
    except Exception as e:
        print(f"Failed to initialize OTLP tracing: {e}")
        return TracerProvider()


def get_tracer(name: str) -> trace.Tracer:
    """Get tracer instance.

    Args:
        name: Tracer name (typically __name__).

    Returns:
        Configured tracer.
    """
    return trace.get_tracer(name)


class RAGSpans:
    """Context managers for RAG operation spans."""

    def __init__(self):
        """Initialize span helpers."""
        self.tracer = get_tracer(__name__)

    def retrieval_span(
        self,
        query_id: str,
        mode: str,
        top_k: int,
    ):
        """Create span for retrieval operation.

        Args:
            query_id: Request identifier.
            mode: Search mode.
            top_k: Number of results.

        Returns:
            Span context manager.
        """
        with self.tracer.start_as_current_span("retrieval") as span:
            span.set_attribute("query_id", query_id)
            span.set_attribute("mode", mode)
            span.set_attribute("top_k", top_k)
            yield span

    def reranking_span(
        self,
        query_id: str,
        num_candidates: int,
    ):
        """Create span for reranking operation.

        Args:
            query_id: Request identifier.
            num_candidates: Number of candidates.

        Returns:
            Span context manager.
        """
        with self.tracer.start_as_current_span("reranking") as span:
            span.set_attribute("query_id", query_id)
            span.set_attribute("num_candidates", num_candidates)
            yield span

    def generation_span(
        self,
        query_id: str,
        llm_model: str,
    ):
        """Create span for generation operation.

        Args:
            query_id: Request identifier.
            llm_model: LLM model name.

        Returns:
            Span context manager.
        """
        with self.tracer.start_as_current_span("generation") as span:
            span.set_attribute("query_id", query_id)
            span.set_attribute("llm_model", llm_model)
            yield span
