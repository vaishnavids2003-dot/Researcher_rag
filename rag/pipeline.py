
from rag.document_loader import load_document
from rag.chunker import chunk_documents
from rag.embedder import embed_texts
from rag.vector_store import VectorStore
from rag.strategies import retrieve_with_strategy
from rag.generator import generate_answer


def ingest_document(
    file_path: str,
    vector_store: VectorStore,
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> dict:
    """Load, chunk, embed, and store one document."""

    documents = load_document(file_path)

    chunks = chunk_documents(
        documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise ValueError("Document produced no text chunks")

    embeddings = embed_texts(
        [chunk["text"] for chunk in chunks]
    )

    document_id = chunks[0]["metadata"]["source"]

    # Replace older chunks when re-indexing this document.
    vector_store.delete_document(document_id)

    stored_count = vector_store.add_chunks(
        chunks=chunks,
        embeddings=embeddings,
        document_id=document_id,
    )

    return {
        "document_id": document_id,
        "chunk_count": stored_count,
    }


def answer_question(
    query: str,
    vector_store: VectorStore,
    top_k: int = 5,
    document_id: str | None = None,
    strategy: str = "similarity",
) -> dict:
    """Retrieve evidence with a selected strategy and generate an answer."""

    if not query.strip():
        raise ValueError("Query cannot be empty")

    chunks = retrieve_with_strategy(
        query=query,
        vector_store=vector_store,
        strategy=strategy,
        top_k=top_k,
        document_id=document_id,
    )

    answer = generate_answer(
        query=query,
        chunks=chunks,
    )

    return {
        "answer": answer,
        "sources": chunks,
    }
