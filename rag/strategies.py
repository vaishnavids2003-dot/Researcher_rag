
from collections.abc import Callable

from rag.query_expansion import generate_alternative_queries
from rag.retriever import retrieve_chunks
from rag.vector_store import VectorStore


def _similarity(
    query: str,
    vector_store: VectorStore,
    top_k: int = 5,
    document_id: str | None = None,
    **kwargs,
) -> list[dict]:
    """Retrieve chunks using the original query."""
    return retrieve_chunks(
        query=query,
        vector_store=vector_store,
        top_k=top_k,
        document_id=document_id,
    )


def _multiquery(
    query: str,
    vector_store: VectorStore,
    top_k: int = 5,
    document_id: str | None = None,
    **kwargs,
) -> list[dict]:
    """Retrieve chunks using multiple query variations."""

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    num_queries = kwargs.get("num_queries", 3)

    if num_queries <= 0:
        raise ValueError("num_queries must be greater than zero")

    alternative_queries = generate_alternative_queries(
        query=query,
        num_queries=num_queries,
    )

    # Search the original query first.
    queries = [query, *alternative_queries]

    unique_chunks = []
    seen_chunks = set()

    for search_query in queries:
        chunks = retrieve_chunks(
            query=search_query,
            vector_store=vector_store,
            top_k=top_k,
            document_id=document_id,
        )

        for chunk in chunks:
            metadata = chunk.get("metadata", {})

            # Identify the same chunk using its text and metadata.
            key = (
                chunk.get("text", ""),
                repr(sorted(metadata.items())),
            )

            if key not in seen_chunks:
                seen_chunks.add(key)
                unique_chunks.append(chunk)

    # Keep the first top_k unique chunks.
    return unique_chunks[:top_k]


def _contextual(
    query: str,
    vector_store: VectorStore,
    top_k: int = 5,
    document_id: str | None = None,
    **kwargs,
) -> list[dict]:
    """Placeholder for contextual retrieval."""
    raise NotImplementedError(
        "Contextual retrieval will be implemented in the next step."
    )


# Registry of supported retrieval strategies.
STRATEGIES: dict[str, Callable] = {
    "similarity": _similarity,
    "multiquery": _multiquery,
    "contextual": _contextual,
}


def retrieve_with_strategy(
    query: str,
    vector_store: VectorStore,
    strategy: str = "similarity",
    top_k: int = 5,
    document_id: str | None = None,
    **kwargs,
) -> list[dict]:
    """Dispatch retrieval to the selected strategy."""

    if strategy not in STRATEGIES:
        raise ValueError(
            f"Unknown retrieval strategy: {strategy}. "
            f"Choose from: {', '.join(STRATEGIES)}"
        )

    return STRATEGIES[strategy](
        query=query,
        vector_store=vector_store,
        top_k=top_k,
        document_id=document_id,
        **kwargs,
    )
