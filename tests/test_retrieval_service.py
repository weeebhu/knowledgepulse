from knowledgepulse.embeddings.provider import EmbeddingProvider
from knowledgepulse.models.embedding import EmbeddedChunk
from knowledgepulse.retrieval.in_memory_vector_store import InMemoryVectorStore
from knowledgepulse.retrieval.service import retrieve


def test_retrieve_embeds_query_and_returns_relevant_chunks() -> None:
    """Retrieval should embed a query and return similar chunks."""

    class FakeEmbeddingProvider(EmbeddingProvider):
        model_name = "fake-model"

        def embed_many(self, texts: list[str]) -> list[list[float]]:
            """Return deterministic vectors for testing."""

            vectors = {
                "deployment approval": [1.0, 0.0],
                "Production deployments require approval.": [1.0, 0.0],
                "Rollback procedures must be documented.": [0.0, 1.0],
            }

            return [vectors[text] for text in texts]

    provider = FakeEmbeddingProvider()

    store = InMemoryVectorStore()

    store.add(
        [
            EmbeddedChunk(
                chunk_id="DOC-001-P001-C000",
                document_id="DOC-001",
                page_number=1,
                chunk_index=0,
                text="Production deployments require approval.",
                vector=[1.0, 0.0],
                model_name="fake-model",
            ),
            EmbeddedChunk(
                chunk_id="DOC-001-P001-C001",
                document_id="DOC-001",
                page_number=1,
                chunk_index=1,
                text="Rollback procedures must be documented.",
                vector=[0.0, 1.0],
                model_name="fake-model",
            ),
        ]
    )

    results = retrieve(
        query="deployment approval",
        provider=provider,
        vector_store=store,
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == "DOC-001-P001-C000"
