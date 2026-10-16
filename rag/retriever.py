
from rag.embedder import embed_query
from rag.vector_store import VectorStore


def retrieve_chunks(
    query: str,
    vector_store: VectorStore,
    top_k: int = 5,
    document_id: str | None = None,
) -> list[dict]:
    """Retrieve the most relevant document chunks for a question."""

    if not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    query_embedding = embed_query(query)

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k,
        document_id=document_id,
    )

    return results
