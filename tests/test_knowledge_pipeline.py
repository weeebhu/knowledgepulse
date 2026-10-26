from pathlib import Path

import pytest

from knowledgepulse.embeddings.provider import EmbeddingProvider
from knowledgepulse.models.document import DocumentPage, IngestedDocument
from knowledgepulse.pipeline.knowledge_pipeline import KnowledgePipeline


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for pipeline tests."""

    model_name = "fake-test-model"

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        """Return deterministic vectors based on text content."""

        vectors = []

        for text in texts:
            if "deployment" in text.lower():
                vectors.append([1.0, 0.0])
            elif "rollback" in text.lower():
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])

        return vectors


class FailingEmbeddingProvider(FakeEmbeddingProvider):
    """Embedding provider that can simulate an upstream failure."""

    def __init__(self) -> None:
        self.fail = False

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("Embedding service unavailable")
        return super().embed_many(texts)


def test_knowledge_pipeline_indexes_and_searches_document() -> None:
    """The pipeline should index a document and retrieve relevant chunks."""

    provider = FakeEmbeddingProvider()
    pipeline = KnowledgePipeline(embedding_provider=provider, max_chunk_size=100)
    document = IngestedDocument(
        document_id="DOC-001",
        source_path="datasets/golden/DOC-001.pdf",
        pages=[
            DocumentPage(page_number=1, text="Production deployments require approval."),
            DocumentPage(page_number=2, text="Rollback procedures must be documented."),
        ],
    )

    pipeline.index_document(document)
    results = pipeline.search(query="What are the deployment requirements?", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.document_id == "DOC-001"
    assert results[0].chunk.page_number == 1
    assert "deployment" in results[0].chunk.text.lower()


@pytest.mark.parametrize("max_chunk_size", [0, -1])
def test_knowledge_pipeline_rejects_invalid_chunk_size(max_chunk_size: int) -> None:
    """The pipeline should reject non-positive chunk sizes."""

    with pytest.raises(ValueError):
        KnowledgePipeline(
            embedding_provider=FakeEmbeddingProvider(),
            max_chunk_size=max_chunk_size,
        )


def test_knowledge_pipeline_indexes_multiple_documents() -> None:
    """The pipeline should index and search across multiple documents."""

    pipeline = KnowledgePipeline(
        embedding_provider=FakeEmbeddingProvider(),
        max_chunk_size=100,
    )
    documents = [
        IngestedDocument(
            document_id="DOC-001",
            source_path="datasets/golden/DOC-001.pdf",
            pages=[
                DocumentPage(page_number=1, text="Production deployments require approval.")
            ],
        ),
        IngestedDocument(
            document_id="DOC-002",
            source_path="datasets/golden/DOC-002.pdf",
            pages=[
                DocumentPage(page_number=1, text="Rollback procedures must be documented.")
            ],
        ),
    ]

    pipeline.index_documents(documents)
    results = pipeline.search(query="What are the rollback requirements?", top_k=1)

    assert len(results) == 1
    assert results[0].chunk.document_id == "DOC-002"
    assert "rollback" in results[0].chunk.text.lower()


def test_indexing_same_document_replaces_existing_chunks() -> None:
    provider = FakeEmbeddingProvider()
    pipeline = KnowledgePipeline(embedding_provider=provider, max_chunk_size=100)
    document = IngestedDocument(
        document_id="DOC-001",
        source_path=Path("DOC-001.pdf"),
        pages=[DocumentPage(page_number=1, text="First version of the document.")],
    )
    updated_document = IngestedDocument(
        document_id="DOC-001",
        source_path=Path("DOC-001.pdf"),
        pages=[DocumentPage(page_number=1, text="Updated version of the document.")],
    )

    pipeline.index_document(document)
    pipeline.index_document(updated_document)
    results = pipeline.search(query="Updated version", top_k=10)

    assert len(results) == 1
    assert results[0].chunk.text == "Updated version of the document."


def test_failed_reindex_preserves_the_previous_document_index() -> None:
    provider = FailingEmbeddingProvider()
    pipeline = KnowledgePipeline(embedding_provider=provider, max_chunk_size=100)
    original = IngestedDocument(
        document_id="DOC-001",
        source_path=Path("DOC-001.pdf"),
        pages=[DocumentPage(page_number=1, text="Original deployment policy.")],
    )
    replacement = IngestedDocument(
        document_id="DOC-001",
        source_path=Path("DOC-001.pdf"),
        pages=[DocumentPage(page_number=1, text="Replacement deployment policy.")],
    )

    pipeline.index_document(original)
    provider.fail = True

    with pytest.raises(RuntimeError, match="Embedding service unavailable"):
        pipeline.index_document(replacement)

    provider.fail = False
    results = pipeline.search(query="deployment policy", top_k=10)

    assert len(results) == 1
    assert results[0].chunk.text == "Original deployment policy."
