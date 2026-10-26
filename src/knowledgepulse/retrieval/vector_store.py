from abc import ABC, abstractmethod

from knowledgepulse.models.embedding import EmbeddedChunk
from knowledgepulse.models.retrieval import RetrievalResult


class VectorStore(ABC):
    """Abstract interface for storing and searching embedded chunks."""

    @abstractmethod
    def add(self, chunks: list[EmbeddedChunk]) -> None:
        """Store embedded chunks."""

    @abstractmethod
    def replace_document(self, document_id: str, chunks: list[EmbeddedChunk]) -> None:
        """Atomically replace all indexed chunks belonging to a document."""

    @abstractmethod
    def get_by_ids(self, chunk_ids: list[str]) -> list[EmbeddedChunk]:
        """Resolve chunk IDs while preserving the caller's requested order."""

    @abstractmethod
    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all chunks belonging to a document."""

    @abstractmethod
    def search(self, query_vector: list[float], top_k: int) -> list[RetrievalResult]:
        """Return the most relevant chunks with similarity scores."""
