
def chunk_documents(
    documents: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> list[dict]:
    """
    Split extracted documents into overlapping text chunks.

    Each chunk retains its original source metadata.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be smaller than chunk_size"
        )

    chunks = []

    for document in documents:
        text = document["text"].strip()
        metadata = document["metadata"]

        if not text:
            continue

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    {
                        "text": chunk_text,
                        "metadata": metadata.copy(),
                    }
                )

            if end == len(text):
                break

            start = end - chunk_overlap

    return chunks
