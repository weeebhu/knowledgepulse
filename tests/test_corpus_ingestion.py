from pathlib import Path

import pytest

from knowledgepulse.ingestion.corpus_ingestion import ingest_directory
from knowledgepulse.models.corpus import DocumentCorpus


def test_ingest_directory_returns_corpus() -> None:
    """A directory of PDFs should be ingested into a document corpus."""

    directory = Path("datasets/golden")

    corpus = ingest_directory(directory)

    assert isinstance(corpus, DocumentCorpus)
    assert corpus.document_count == 4


def test_ingest_directory_rejects_missing_directory() -> None:
    """A missing directory should raise a clear error."""

    directory = Path("datasets/does-not-exist")

    with pytest.raises(FileNotFoundError, match="Directory does not exist"):
        ingest_directory(directory)


def test_ingest_directory_returns_documents_in_stable_order() -> None:
    """Documents should be returned in deterministic filename order."""

    directory = Path("datasets/golden")

    corpus = ingest_directory(directory)

    assert corpus.document_ids == [
        "DOC-001",
        "DOC-002",
        "DOC-003",
        "DOC-004",
    ]
