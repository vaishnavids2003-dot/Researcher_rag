import os
import pytest
from pathlib import Path

from rag.pipeline import ingest_document, answer_question
from rag.vector_store import VectorStore

@pytest.mark.skipif(
    os.getenv("RUN_LIVE_GEMINI_TESTS") != "1",
    reason="Live Gemini integration test is opt-in",
)

def test_real_rag_pipeline():
    project_root = Path(__file__).resolve().parents[1]

    document_path = (
        project_root
        / "data"
        / "language_models_are_unsupervised_multitask_learners.pdf"
    )

    assert document_path.is_file(), (
        f"PDF not found: {document_path}"
    )

    store = VectorStore(
        persist_directory=str(project_root / "test_chroma_db"),
        collection_name="integration_test",
    )

    # Step 1: Extract, chunk, embed, and store the PDF.
    ingestion = ingest_document(str(document_path), store)

    assert ingestion["chunk_count"] > 0
    print(f"\nChunks indexed: {ingestion['chunk_count']}")

    # Step 2: Retrieve relevant chunks and generate an answer.
    question = "What is the main idea behind language models as unsupervised multitask learners?"

    result = answer_question(
        query=question,
        vector_store=store,
        top_k=3,
    )

    assert result["answer"].strip()
    assert result["sources"]

    print("\nQuestion:", question)
    print("\nGenerated answer:", result["answer"])
    print("\nRetrieved sources:")

    for source in result["sources"]:
        print("\nMetadata:", source["metadata"])
        print("Text:", source["text"][:300])
        print("Distance:", source["distance"])


if __name__ == "__main__":
    test_real_rag_pipeline()
