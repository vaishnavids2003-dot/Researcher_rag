
import pytest

from rag import retriever


class FakeVectorStore:
    """A small substitute for ChromaDB during unit tests."""

    def __init__(self):
        self.received_arguments = None

    def search(
        self,
        query_embedding,
        top_k=5,
        document_id=None,
    ):
        self.received_arguments = {
            "query_embedding": query_embedding,
            "top_k": top_k,
            "document_id": document_id,
        }

        return [
            {
                "text": "Python is a programming language.",
                "metadata": {"source": "python.txt"},
                "distance": 0.05,
            }
        ]


def test_retrieve_chunks(monkeypatch):
    store = FakeVectorStore()

    def fake_embed_query(query):
        assert query == "What is Python?"
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(
        retriever, "embed_query", fake_embed_query
    )

    results = retriever.retrieve_chunks(
        "What is Python?",
        store,
        top_k=3,
    )

    assert len(results) == 1
    assert results[0]["text"] == "Python is a programming language."
    assert store.received_arguments["query_embedding"] == [
        0.1, 0.2, 0.3
    ]
    assert store.received_arguments["top_k"] == 3


def test_retrieve_chunks_with_document_filter(monkeypatch):
    store = FakeVectorStore()

    monkeypatch.setattr(
        retriever,
        "embed_query",
        lambda query: [0.1, 0.2, 0.3],
    )

    retriever.retrieve_chunks(
        "What is Python?",
        store,
        document_id="python_doc",
    )

    assert store.received_arguments["document_id"] == "python_doc"


def test_empty_query():
    store = FakeVectorStore()

    with pytest.raises(ValueError, match="Query cannot be empty"):
        retriever.retrieve_chunks("   ", store)


def test_invalid_top_k():
    store = FakeVectorStore()

    with pytest.raises(ValueError, match="top_k"):
        retriever.retrieve_chunks("What is Python?", store, top_k=0)
