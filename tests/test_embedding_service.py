from knowledgepulse.embeddings.provider import EmbeddingProvider
from knowledgepulse.embeddings.service import embed_chunks
from knowledgepulse.models.chunk import TextChunk


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for tests."""

    model_name = "fake-embedding-model"

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic vectors for testing."""
        return [[float(len(text)), 1.0, 0.5] for text in texts]


def test_embed_chunks_preserves_chunk_provenance() -> None:
    """Embedding chunks should preserve their original provenance."""

    chunks = [
        TextChunk(
            chunk_id="DOC-001-P001-C000",
            document_id="DOC-001",
            page_number=1,
            chunk_index=0,
            text="Production deployments require approval.",
        ),
        TextChunk(
            chunk_id="DOC-001-P002-C000",
            document_id="DOC-001",
            page_number=2,
            chunk_index=0,
            text="Rollback procedures must be documented.",
        ),
    ]

    provider = FakeEmbeddingProvider()

    embedded_chunks = embed_chunks(
        chunks=chunks,
        provider=provider,
    )

    assert len(embedded_chunks) == 2

    assert embedded_chunks[0].chunk_id == "DOC-001-P001-C000"
    assert embedded_chunks[0].document_id == "DOC-001"
    assert embedded_chunks[0].page_number == 1
    assert embedded_chunks[0].chunk_index == 0
    assert embedded_chunks[0].text == "Production deployments require approval."

    assert embedded_chunks[1].chunk_id == "DOC-001-P002-C000"
    assert embedded_chunks[1].page_number == 2

    assert embedded_chunks[0].vector == [
        float(len("Production deployments require approval.")),
        1.0,
        0.5,
    ]

    assert embedded_chunks[0].model_name == "fake-embedding-model"


def test_embed_chunks_sends_all_texts_to_provider_in_one_batch() -> None:
    """Embedding should process multiple chunks through a batch provider call."""

    class BatchRecordingProvider(EmbeddingProvider):
        model_name = "batch-test-model"

        def __init__(self) -> None:
            self.received_texts: list[str] | None = None

        def embed_many(self, texts: list[str]) -> list[list[float]]:
            """Record received texts and return deterministic vectors."""
            self.received_texts = texts

            return [[float(index), float(len(text))] for index, text in enumerate(texts)]

    chunks = [
        TextChunk(
            chunk_id="DOC-001-P001-C000",
            document_id="DOC-001",
            page_number=1,
            chunk_index=0,
            text="First chunk.",
        ),
        TextChunk(
            chunk_id="DOC-001-P001-C001",
            document_id="DOC-001",
            page_number=1,
            chunk_index=1,
            text="Second chunk.",
        ),
    ]

    provider = BatchRecordingProvider()

    embedded_chunks = embed_chunks(
        chunks=chunks,
        provider=provider,
    )

    assert provider.received_texts == [
        "First chunk.",
        "Second chunk.",
    ]

    assert len(embedded_chunks) == 2
    assert embedded_chunks[0].vector == [0.0, 12.0]
    assert embedded_chunks[1].vector == [1.0, 13.0]
