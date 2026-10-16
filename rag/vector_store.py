
from pathlib import Path

import chromadb


class VectorStore:
    """Store and search document chunks using ChromaDB."""

    def __init__(
        self,
        persist_directory: str = "chroma_db",
        collection_name: str = "documents",
    ):
        self.client = chromadb.PersistentClient(
            path=str(Path(persist_directory))
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: list[dict],
        embeddings: list[list[float]],
        document_id: str | None = None,
    ) -> int:
        """Store chunks, their embeddings, and their metadata."""

        if len(chunks) != len(embeddings):
            raise ValueError(
                "The number of chunks must match the number of embeddings"
            )

        if not chunks:
            return 0

        if not document_id:
            document_id = chunks[0]["metadata"]["source"]

        ids = []
        texts = []
        metadatas = []

        for index, chunk in enumerate(chunks):
            chunk_id = f"{document_id}:{index}"

            metadata = chunk["metadata"].copy()
            metadata["document_id"] = document_id
            metadata["chunk_index"] = index

            ids.append(chunk_id)
            texts.append(chunk["text"])
            metadatas.append(metadata)

        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        return len(chunks)

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        document_id: str | None = None,
    ) -> list[dict]:
        """Return the most similar chunks for a query vector."""

        if not query_embedding:
            raise ValueError("Query embedding cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_args = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }

        if document_id:
            query_args["where"] = {"document_id": document_id}

        results = self.collection.query(**query_args)

        matches = []

        for text, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            matches.append(
                {
                    "text": text,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return matches

    def delete_document(self, document_id: str) -> int:
        """Delete every stored chunk belonging to one document."""

        existing = self.collection.get(
            where={"document_id": document_id},
            include=["metadatas"],
        )

        ids = existing["ids"]

        if not ids:
            return 0

        self.collection.delete(ids=ids)

        return len(ids)
