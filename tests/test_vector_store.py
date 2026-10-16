
import pytest

from rag.vector_store import VectorStore


@pytest.fixture
def store(tmp_path):
    """Create a fresh temporary vector database for each test."""
    return VectorStore(
        persist_directory=str(tmp_path / "test_chroma_db")
    )


@pytest.fixture
def sample_chunks():
    return [
        {
            "text": "Python is a programming language.",
            "metadata": {"source": "python.txt"},
        },
        {
            "text": "The capital of France is Paris.",
            "metadata": {"source": "python.txt"},
        },
    ]


@pytest.fixture
def sample_embeddings():
    return [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]


def test_add_chunks(store, sample_chunks, sample_embeddings):
    count = store.add_chunks(sample_chunks, sample_embeddings)

    assert count == 2
    assert store.collection.count() == 2


def test_search_returns_closest_chunk(
    store, sample_chunks, sample_embeddings
):
    store.add_chunks(sample_chunks, sample_embeddings)

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0]["text"] == "Python is a programming language."
    assert results[0]["metadata"]["source"] == "python.txt"
    assert results[0]["distance"] < 0.01


def test_search_can_filter_by_document(
    store, sample_chunks, sample_embeddings
):
    store.add_chunks(
        sample_chunks,
        sample_embeddings,
        document_id="python_doc",
    )

    results = store.search(
        query_embedding=[1.0, 0.0, 0.0],
        top_k=5,
        document_id="python_doc",
    )

    assert len(results) == 2


def test_delete_document(store, sample_chunks, sample_embeddings):
    store.add_chunks(
        sample_chunks,
        sample_embeddings,
        document_id="python_doc",
    )

    deleted_count = store.delete_document("python_doc")

    assert deleted_count == 2
    assert store.collection.count() == 0


def test_empty_chunks(store):
    count = store.add_chunks([], [])

    assert count == 0


def test_mismatched_chunk_and_embedding_counts(
    store, sample_chunks, sample_embeddings
):
    with pytest.raises(ValueError, match="must match"):
        store.add_chunks(sample_chunks, sample_embeddings[:1])


def test_invalid_top_k(store):
    with pytest.raises(ValueError, match="top_k"):
        store.search([1.0, 0.0, 0.0], top_k=0)


def test_empty_query_embedding(store):
    with pytest.raises(ValueError, match="cannot be empty"):
        store.search([], top_k=1)
