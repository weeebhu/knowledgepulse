import math

from knowledgepulse.models.embedding import EmbeddedChunk
from knowledgepulse.models.retrieval import RetrievalResult
from knowledgepulse.retrieval.vector_store import VectorStore


class InMemoryVectorStore(VectorStore):
    """In-memory vector store using cosine similarity for retrieval."""

    def __init__(self) -> None:
        self._chunks: list[EmbeddedChunk] = []

    def add(self, chunks: list[EmbeddedChunk]) -> None:
        self._validate_unique_chunk_ids(chunks)
        existing_ids = {chunk.chunk_id for chunk in self._chunks}
        incoming_ids = {chunk.chunk_id for chunk in chunks}
        if existing_ids.intersection(incoming_ids):
            raise ValueError("Chunk ID already exists in the vector store")
        self._chunks.extend(chunks)

    def replace_document(self, document_id: str, chunks: list[EmbeddedChunk]) -> None:
        if not document_id.strip():
            raise ValueError("document_id cannot be empty")
        self._validate_unique_chunk_ids(chunks)
        if any(chunk.document_id != document_id for chunk in chunks):
            raise ValueError("All replacement chunks must belong to the target document")

        replacement_ids = {chunk.chunk_id for chunk in chunks}
        retained = [chunk for chunk in self._chunks if chunk.document_id != document_id]
        retained_ids = {chunk.chunk_id for chunk in retained}
        if retained_ids.intersection(replacement_ids):
            raise ValueError("Chunk ID already exists for another document")
        self._chunks = [*retained, *chunks]

    @staticmethod
    def _validate_unique_chunk_ids(chunks: list[EmbeddedChunk]) -> None:
        ids = [chunk.chunk_id for chunk in chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("Chunk ID already exists")

    def get_by_ids(self, chunk_ids: list[str]) -> list[EmbeddedChunk]:
        """Resolve IDs in requested order; fail loudly on index inconsistency."""
        by_id = {chunk.chunk_id: chunk for chunk in self._chunks}
        missing = [chunk_id for chunk_id in chunk_ids if chunk_id not in by_id]
        if missing:
            raise KeyError(f"Chunks not found in vector store: {', '.join(missing)}")
        return [by_id[chunk_id] for chunk_id in chunk_ids]

    def search(self, query_vector: list[float], top_k: int) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        scored = [
            (self._cosine_similarity(query_vector, chunk.vector), chunk)
            for chunk in self._chunks
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievalResult(chunk=chunk, score=score)
            for score, chunk in scored[:top_k]
        ]

    @staticmethod
    def _cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
        dot = sum(a * b for a, b in zip(vector_a, vector_b, strict=True))
        magnitude_a = math.sqrt(sum(value * value for value in vector_a))
        magnitude_b = math.sqrt(sum(value * value for value in vector_b))
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        return dot / (magnitude_a * magnitude_b)

    def delete_by_document_id(self, document_id: str) -> None:
        self._chunks = [chunk for chunk in self._chunks if chunk.document_id != document_id]
