
from unittest.mock import Mock

import pytest

import rag.generator as generator


def test_generate_answer(monkeypatch):
    captured = {}

    def fake_gemini(prompt, system_prompt, model=None):
        captured["prompt"] = prompt
        captured["system_prompt"] = system_prompt
        captured["model"] = model

        return "Python is a programming language."

    monkeypatch.setattr(
        generator,
        "generate_with_gemini",
        fake_gemini,
    )

    chunks = [
        {
            "text": "Python is a programming language.",
            "metadata": {
                "source": "python.pdf",
                "page": 1,
            },
        }
    ]

    answer = generator.generate_answer(
        query="What is Python?",
        chunks=chunks,
    )

    assert answer == "Python is a programming language."
    assert "What is Python?" in captured["prompt"]
    assert "Python is a programming language." in captured["prompt"]
    assert "python.pdf" in captured["prompt"]
    assert "page: 1" in captured["prompt"]
    assert "untrusted data" in captured["system_prompt"]


def test_empty_query():
    with pytest.raises(ValueError, match="Query cannot be empty"):
        generator.generate_answer(
            query="   ",
            chunks=[{"text": "Some content", "metadata": {}}],
        )


def test_no_retrieved_chunks(monkeypatch):
    def unexpected_gemini(*args, **kwargs):
        pytest.fail("LLM should not be called when there are no chunks")

    monkeypatch.setattr(
        generator,
        "generate_with_gemini",
        unexpected_gemini,
    )

    answer = generator.generate_answer(
        query="What is Python?",
        chunks=[],
    )

    assert answer == (
        "I couldn't find relevant information in the uploaded documents."
    )


def test_empty_model_response(monkeypatch):
    monkeypatch.setattr(
        generator,
        "generate_with_gemini",
        Mock(return_value="   "),
    )

    with pytest.raises(
        ValueError,
        match="language model returned an empty answer",
    ):
        generator.generate_answer(
            query="What is Python?",
            chunks=[
                {
                    "text": "Python is a programming language.",
                    "metadata": {},
                }
            ],
        )
