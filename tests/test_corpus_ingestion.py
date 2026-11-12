from pathlib import Path

import pytest

from knowledgepulse.ingestion.corpus_ingestion import ingest_directory
from knowledgepulse.models.corpus import DocumentCorpus


def test_ingest_directory_returns_corpus() -> None:
    """A directory of supported documents should be ingested into a corpus."""
    corpus = ingest_directory(Path("datasets/golden"))

    assert isinstance(corpus, DocumentCorpus)
    assert corpus.document_count == 8


def test_ingest_directory_rejects_missing_directory() -> None:
    """A missing directory should raise a clear error."""
    with pytest.raises(FileNotFoundError, match="Directory does not exist"):
        ingest_directory(Path("datasets/does-not-exist"))


def test_ingest_directory_returns_documents_in_stable_order() -> None:
    """Documents should be returned in deterministic filename order."""
    corpus = ingest_directory(Path("datasets/golden"))

    assert corpus.document_ids == [
        "DOC-001",
        "DOC-002",
        "DOC-003",
        "DOC-004",
        "DOC-005",
        "DOC-006",
        "DOC-007",
        "DOC-008",
    ]
