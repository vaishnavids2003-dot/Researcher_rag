
import pytest

from rag import strategies
from rag.strategies import STRATEGIES, retrieve_with_strategy


def test_strategy_registry_contains_expected_strategies():
    assert set(STRATEGIES) == {
        "similarity",
        "multiquery",
        "contextual",
    }


def test_similarity_strategy_uses_existing_retriever(monkeypatch):
    expected_chunks = [
        {
            "text": "Python is a programming language.",
            "metadata": {},
        }
    ]

    def fake_retrieve_chunks(
        query, vector_store, top_k=5, document_id=None
    ):
        assert query == "What is Python?"
        assert top_k == 3
        assert document_id == "python.pdf"
        return expected_chunks

    monkeypatch.setattr(
        strategies, "retrieve_chunks", fake_retrieve_chunks
    )

    result = retrieve_with_strategy(
        query="What is Python?",
        vector_store=object(),
        strategy="similarity",
        top_k=3,
        document_id="python.pdf",
    )

    assert result == expected_chunks


def test_unknown_strategy_raises_error():
    with pytest.raises(ValueError, match="Unknown retrieval strategy"):
        retrieve_with_strategy(
            query="What is Python?",
            vector_store=object(),
            strategy="unknown",
        )


def test_multiquery_retrieves_and_deduplicates_chunks(monkeypatch):
    chunk_a = {
        "text": "Python supports machine learning.",
        "metadata": {"source": "guide.txt"},
    }

    chunk_b = {
        "text": "Python has many AI libraries.",
        "metadata": {"source": "guide.txt"},
    }

    monkeypatch.setattr(
        strategies,
        "generate_alternative_queries",
        lambda query, num_queries=3: [
            "Python and machine learning",
            "Python AI libraries",
        ],
    )

    calls = []

    def fake_retrieve_chunks(
        query, vector_store, top_k=5, document_id=None
    ):
        calls.append(
            {
                "query": query,
                "top_k": top_k,
                "document_id": document_id,
            }
        )

        if query == "What is Python useful for?":
            return [chunk_a]

        if query == "Python and machine learning":
            return [chunk_a, chunk_b]

        return [chunk_b]

    monkeypatch.setattr(
        strategies, "retrieve_chunks", fake_retrieve_chunks
    )

    result = retrieve_with_strategy(
        query="What is Python useful for?",
        vector_store=object(),
        strategy="multiquery",
        top_k=5,
        document_id="guide.txt",
    )

    assert result == [chunk_a, chunk_b]
    assert len(calls) == 3

    assert all(
        call["document_id"] == "guide.txt"
        for call in calls
    )

    assert all(call["top_k"] == 5 for call in calls)


def test_multiquery_rejects_empty_query():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        retrieve_with_strategy(
            query="   ",
            vector_store=object(),
            strategy="multiquery",
        )


def test_multiquery_rejects_invalid_top_k():
    with pytest.raises(ValueError, match="top_k"):
        retrieve_with_strategy(
            query="What is Python?",
            vector_store=object(),
            strategy="multiquery",
            top_k=0,
        )


def test_multiquery_rejects_invalid_num_queries():
    with pytest.raises(ValueError, match="num_queries"):
        retrieve_with_strategy(
            query="What is Python?",
            vector_store=object(),
            strategy="multiquery",
            num_queries=0,
        )
