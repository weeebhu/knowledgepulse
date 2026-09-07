from sentence_transformers import SentenceTransformer

from knowledgepulse.embeddings.provider import EmbeddingProvider


class SentenceTransformerProvider(EmbeddingProvider):
    """Embedding provider backed by SentenceTransformers."""

    model_name = "BAAI/bge-small-en-v1.5"

    def __init__(self) -> None:
        """Load the embedding model."""
        self.model = SentenceTransformer(self.model_name)

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Convert multiple texts into embedding vectors."""
        vectors = self.model.encode(texts)

        return vectors.tolist()
