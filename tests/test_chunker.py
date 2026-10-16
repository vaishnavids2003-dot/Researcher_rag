
import pytest

from rag.chunker import chunk_documents


def test_chunk_documents():
    documents = [
        {
            "text": "abcdefghijklmnopqrstuvwxyz",
            "metadata": {"source": "test.txt"},
        }
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=10,
        chunk_overlap=3,
    )

    assert len(chunks) == 4
    assert chunks[0]["text"] == "abcdefghij"
    assert chunks[1]["text"] == "hijklmnopq"
    assert chunks[0]["metadata"]["source"] == "test.txt"


def test_empty_document():
    documents = [
        {
            "text": "   ",
            "metadata": {"source": "empty.txt"},
        }
    ]

    assert chunk_documents(documents) == []


def test_invalid_chunk_size():
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_documents([], chunk_size=0)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError, match="smaller"):
        chunk_documents(
            [],
            chunk_size=10,
            chunk_overlap=10,
        )


def test_no_overlap():
    documents = [
        {
            "text": "abcdefghijkl",
            "metadata": {"source": "test.txt"},
        }
    ]

    chunks = chunk_documents(
        documents,
        chunk_size=5,
        chunk_overlap=0,
    )

    assert [chunk["text"] for chunk in chunks] == [
        "abcde",
        "fghij",
        "kl",
    ]
