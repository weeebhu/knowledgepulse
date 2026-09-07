from pathlib import Path

import pytest

from knowledgepulse.ingestion.chunking import chunk_document
from knowledgepulse.models.document import DocumentPage, IngestedDocument


def test_chunk_document_preserves_page_provenance() -> None:
    """Chunking should preserve the source document and page."""

    document = IngestedDocument(
        document_id="DOC-001",
        source_path=Path("datasets/golden/DOC-001.pdf"),
        pages=[
            DocumentPage(
                page_number=1,
                text="First page contains deployment instructions.",
            ),
            DocumentPage(
                page_number=2,
                text="Second page contains rollback instructions.",
            ),
        ],
    )

    chunks = chunk_document(
        document=document,
        max_chunk_size=100,
    )

    assert len(chunks) == 2

    assert chunks[0].document_id == "DOC-001"
    assert chunks[0].page_number == 1
    assert chunks[0].text == "First page contains deployment instructions."

    assert chunks[1].document_id == "DOC-001"
    assert chunks[1].page_number == 2
    assert chunks[1].text == "Second page contains rollback instructions."


def test_chunk_document_splits_large_page_into_multiple_chunks() -> None:
    """A page larger than the chunk size should produce multiple chunks."""

    document = IngestedDocument(
        document_id="DOC-001",
        source_path=Path("datasets/golden/DOC-001.pdf"),
        pages=[
            DocumentPage(
                page_number=1,
                text=(
                    "Production deployments require approval. "
                    "Rollback procedures must be documented. "
                    "Database backups must be verified."
                ),
            ),
        ],
    )

    chunks = chunk_document(
        document=document,
        max_chunk_size=50,
    )

    assert len(chunks) > 1

    assert all(chunk.document_id == "DOC-001" for chunk in chunks)
    assert all(chunk.page_number == 1 for chunk in chunks)
    assert all(len(chunk.text) <= 50 for chunk in chunks)


def test_chunk_document_splits_words_larger_than_max_chunk_size() -> None:
    """A single oversized word should still respect the chunk size limit."""

    document = IngestedDocument(
        document_id="DOC-001",
        source_path=Path("datasets/golden/DOC-001.pdf"),
        pages=[
            DocumentPage(
                page_number=1,
                text="supercalifragilisticexpialidocious",
            ),
        ],
    )

    chunks = chunk_document(
        document=document,
        max_chunk_size=10,
    )

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 10 for chunk in chunks)
    assert "".join(chunk.text for chunk in chunks) == ("supercalifragilisticexpialidocious")


def test_chunk_document_generates_deterministic_chunk_ids() -> None:
    """Chunk IDs and indexes should be stable and deterministic."""

    document = IngestedDocument(
        document_id="DOC-001",
        source_path=Path("datasets/golden/DOC-001.pdf"),
        pages=[
            DocumentPage(
                page_number=1,
                text=(
                    "Production deployments require approval. "
                    "Rollback procedures must be documented. "
                    "Database backups must be verified."
                ),
            ),
        ],
    )

    chunks = chunk_document(
        document=document,
        max_chunk_size=50,
    )

    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]

    assert [chunk.chunk_id for chunk in chunks] == [
        "DOC-001-P001-C000",
        "DOC-001-P001-C001",
        "DOC-001-P001-C002",
    ]


def test_chunk_document_resets_chunk_index_for_each_page() -> None:
    """Chunk indexes should restart for each source page."""

    document = IngestedDocument(
        document_id="DOC-001",
        source_path=Path("datasets/golden/DOC-001.pdf"),
        pages=[
            DocumentPage(
                page_number=1,
                text="Alpha bravo charlie delta echo foxtrot.",
            ),
            DocumentPage(
                page_number=2,
                text="Golf hotel india juliet kilo lima.",
            ),
        ],
    )

    chunks = chunk_document(
        document=document,
        max_chunk_size=20,
    )

    page_one_chunks = [chunk for chunk in chunks if chunk.page_number == 1]

    page_two_chunks = [chunk for chunk in chunks if chunk.page_number == 2]

    assert [chunk.chunk_index for chunk in page_one_chunks] == list(range(len(page_one_chunks)))

    assert [chunk.chunk_index for chunk in page_two_chunks] == list(range(len(page_two_chunks)))


@pytest.mark.parametrize("max_chunk_size", [0, -1])
def test_chunk_document_rejects_invalid_max_chunk_size(
    max_chunk_size: int,
) -> None:
    """Chunking should reject non-positive chunk sizes."""

    document = IngestedDocument(
        document_id="DOC-001",
        source_path=Path("datasets/golden/DOC-001.pdf"),
        pages=[
            DocumentPage(
                page_number=1,
                text="Production deployments require approval.",
            ),
        ],
    )

    with pytest.raises(
        ValueError,
        match="max_chunk_size must be greater than 0",
    ):
        chunk_document(
            document=document,
            max_chunk_size=max_chunk_size,
        )
