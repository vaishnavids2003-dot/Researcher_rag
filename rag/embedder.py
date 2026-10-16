
from functools import lru_cache

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""

    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Convert multiple text chunks into embedding vectors."""

    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("Texts cannot contain empty strings")

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    """Convert one user's question into an embedding vector."""

    if not query.strip():
        raise ValueError("Query cannot be empty")

    return embed_texts([query])[0]
