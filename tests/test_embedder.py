
import pytest

from rag.embedder import embed_texts, embed_query


def test_embed_texts():
    texts = [
        "Employees receive annual leave.",
        "The company provides paid vacation.",
    ]

    embeddings = embed_texts(texts)

    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384
    assert all(isinstance(value, float) for value in embeddings[0])


def test_embed_query():
    embedding = embed_query("How much annual leave do employees receive?")

    assert len(embedding) == 384


def test_empty_text_list():
    assert embed_texts([]) == []


def test_empty_text_rejected():
    with pytest.raises(ValueError, match="empty"):
        embed_texts([""])


def test_empty_query_rejected():
    with pytest.raises(ValueError, match="empty"):
        embed_query("   ")
