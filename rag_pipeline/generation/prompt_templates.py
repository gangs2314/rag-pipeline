"""Prompt templates for RAG generation."""

SYSTEM_PROMPT = """You are a helpful assistant answering questions based strictly on the \
provided context.

IMPORTANT RULES:
1. Answer ONLY using information from the provided context chunks.
2. If the context does not contain sufficient information to answer the question, \
respond with:
   "I don't have enough context to answer this question."
3. Cite every source chunk you reference using [chunk_id] format.
4. Be concise and factual.
5. If you must make an inference, clearly state it as such."""

ANSWER_PROMPT = """Context:
{context_block}

Question: {query}

Answer based only on the above context. Cite chunk IDs for all claims."""


def format_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context block.

    Args:
        chunks: List of chunk dicts with 'content', 'metadata', and 'score'.

    Returns:
        Formatted context string with numbered chunks.

    Example output:
        [1] (source=file.pdf, page=3, chunk_id=abc123)
        The content of chunk 1...

        [2] (source=file.pdf, page=4, chunk_id=def456)
        The content of chunk 2...
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        content = chunk.get("content", "")
        metadata = chunk.get("metadata", {})

        source = metadata.get("source_file", "unknown")
        page = metadata.get("page_number", "")
        chunk_id = metadata.get("chunk_id", f"chunk_{i}")

        page_str = f", page={page}" if page else ""
        header = f"[{i}] (source={source}{page_str}, chunk_id={chunk_id})"
        lines.append(header)
        lines.append(content)
        lines.append("")

    return "\n".join(lines).strip()
