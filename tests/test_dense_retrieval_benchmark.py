from knowledgepulse.embeddings.provider import EmbeddingProvider
from knowledgepulse.pipeline.knowledge_pipeline import KnowledgePipeline


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for benchmark integration tests."""

    model_name = "fake-embedding-model"

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


def test_pipeline_search_can_be_adapted_for_benchmark() -> None:
    pipeline = KnowledgePipeline(
        embedding_provider=FakeEmbeddingProvider(),
        max_chunk_size=500,
    )

    def retriever(query: str, top_k: int) -> list[str]:
        results = pipeline.search(query=query, top_k=top_k)

        return [result.chunk.chunk_id for result in results]

    retrieved_chunk_ids = retriever(
        query="production deployment approval",
        top_k=5,
    )

    assert retrieved_chunk_ids == []
