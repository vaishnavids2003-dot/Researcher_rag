
import pytest

from rag import pipeline


class FakeVectorStore:
    """Fake vector store for pipeline unit tests."""

    def __init__(self):
        self.deleted_documents = []
        self.added_chunks = None

    def delete_document(self, document_id):
        self.deleted_documents.append(document_id)

    def add_chunks(self, chunks, embeddings, document_id):
        self.added_chunks = {
            "chunks": chunks,
            "embeddings": embeddings,
            "document_id": document_id,
        }
        return len(chunks)


def test_ingest_document(monkeypatch):
    store = FakeVectorStore()

    documents = [{"text": "Python is useful for AI."}]
    chunks = [
        {
            "text": "Python is useful for AI.",
            "metadata": {"source": "guide.txt"},
        }
    ]

    monkeypatch.setattr(
        pipeline, "load_document", lambda file_path: documents
    )
    monkeypatch.setattr(
        pipeline, "chunk_documents", lambda *args, **kwargs: chunks
    )
    monkeypatch.setattr(
        pipeline, "embed_texts", lambda texts: [[0.1, 0.2, 0.3]]
    )

    result = pipeline.ingest_document("guide.txt", store)

    assert result == {
        "document_id": "guide.txt",
        "chunk_count": 1,
    }
    assert store.deleted_documents == ["guide.txt"]
    assert store.added_chunks["document_id"] == "guide.txt"


def test_ingest_document_with_no_chunks(monkeypatch):
    store = FakeVectorStore()

    monkeypatch.setattr(
        pipeline, "load_document", lambda file_path: []
    )
    monkeypatch.setattr(
        pipeline, "chunk_documents", lambda *args, **kwargs: []
    )

    with pytest.raises(ValueError, match="no text chunks"):
        pipeline.ingest_document("empty.txt", store)


def test_answer_question(monkeypatch):
    store = FakeVectorStore()

    expected_chunks = [
        {
            "text": "Python is useful for AI.",
            "metadata": {"source": "guide.txt"},
            "distance": 0.05,
        }
    ]

    captured = {}

    def fake_retrieve_with_strategy(**kwargs):
        captured.update(kwargs)
        return expected_chunks

    monkeypatch.setattr(
        pipeline,
        "retrieve_with_strategy",
        fake_retrieve_with_strategy,
    )
    monkeypatch.setattr(
        pipeline,
        "generate_answer",
        lambda query, chunks: "Python is useful for AI.",
    )

    result = pipeline.answer_question(
        query="Why use Python for AI?",
        vector_store=store,
        top_k=3,
    )

    assert result["answer"] == "Python is useful for AI."
    assert result["sources"] == expected_chunks
    assert captured["query"] == "Why use Python for AI?"
    assert captured["vector_store"] is store
    assert captured["top_k"] == 3
    assert captured["strategy"] == "similarity"


def test_answer_question_with_empty_query():
    store = FakeVectorStore()

    with pytest.raises(ValueError, match="Query cannot be empty"):
        pipeline.answer_question("   ", store)
