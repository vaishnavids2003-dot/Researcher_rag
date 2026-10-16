
import pytest
from rag import query_expansion


def test_generate_alternative_queries(monkeypatch):
    monkeypatch.setattr(
        query_expansion,
        "generate_with_gemini",
        lambda **kwargs: (
            "What makes Python suitable for AI?\n"
            "What Python features support machine learning?\n"
            "What makes Python suitable for AI?"
        ),
    )

    result = query_expansion.generate_alternative_queries(
        "Why is Python useful for AI?",
        num_queries=3,
    )

    assert result == [
        "What makes Python suitable for AI?",
        "What Python features support machine learning?",
    ]


def test_empty_query():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        query_expansion.generate_alternative_queries("   ")


def test_invalid_num_queries():
    with pytest.raises(ValueError, match="num_queries"):
        query_expansion.generate_alternative_queries(
            "Why is Python useful for AI?",
            num_queries=0,
        )
