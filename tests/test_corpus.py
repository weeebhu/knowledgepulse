from pathlib import Path

import pytest

from knowledgepulse.models.corpus import DocumentCorpus
from knowledgepulse.models.document import DocumentPage, IngestedDocument


def create_document(
    document_id: str,
    page_count: int,
) -> IngestedDocument:
    """Create a test document with the requested number of pages."""

    pages = [
        DocumentPage(
            page_number=page_number,
            text=f"Content for page {page_number}",
        )
        for page_number in range(1, page_count + 1)
    ]

    return IngestedDocument(
        document_id=document_id,
        source_path=Path(f"{document_id}.pdf"),
        pages=pages,
    )


def test_corpus_counts_documents() -> None:
    """Corpus should report the correct number of documents."""

    document_one = create_document("DOC-001", 2)
    document_two = create_document("DOC-002", 3)

    corpus = DocumentCorpus(
        documents=[document_one, document_two],
    )

    assert corpus.document_count == 2


def test_corpus_counts_total_pages() -> None:
    """Corpus should report the total number of pages."""

    document_one = create_document("DOC-001", 2)
    document_two = create_document("DOC-002", 3)

    corpus = DocumentCorpus(
        documents=[document_one, document_two],
    )

    assert corpus.page_count == 5


def test_corpus_returns_document_ids() -> None:
    """Corpus should return all document IDs."""

    document_one = create_document("DOC-001", 1)
    document_two = create_document("DOC-002", 1)

    corpus = DocumentCorpus(
        documents=[document_one, document_two],
    )

    assert corpus.document_ids == [
        "DOC-001",
        "DOC-002",
    ]


def test_corpus_rejects_duplicate_document_ids() -> None:
    """Corpus should reject duplicate document IDs."""

    document_one = create_document("DOC-001", 1)
    document_two = create_document("DOC-001", 2)

    with pytest.raises(ValueError, match="Duplicate document ID"):
        DocumentCorpus(
            documents=[document_one, document_two],
        )
