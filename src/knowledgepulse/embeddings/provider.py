from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Abstract interface for embedding providers."""

    model_name: str

    @abstractmethod
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Convert multiple texts into embedding vectors."""
