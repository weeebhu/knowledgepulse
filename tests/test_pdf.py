from pathlib import Path

import pytest

from knowledgepulse.ingestion.pdf import PDFIngestionError, ingest_pdf


def test_ingest_pdf_extracts_text() -> None:
    """A valid PDF should produce an ingested document."""

    pdf_path = Path("datasets/golden") / "DOC-001_Production_Deployment_Runbook.pdf"

    document = ingest_pdf(
        path=pdf_path,
        document_id="DOC-001",
    )

    assert document.document_id == "DOC-001"
    assert len(document.pages) > 0
    assert "Production Deployment Runbook" in document.text


def test_ingest_pdf_rejects_missing_file() -> None:
    """A missing file should raise a domain-specific error."""

    with pytest.raises(PDFIngestionError, match="File does not exist"):
        ingest_pdf(
            path=Path("does-not-exist.pdf"),
            document_id="DOC-MISSING",
        )


def test_ingest_pdf_rejects_non_pdf(tmp_path: Path) -> None:
    """A non-PDF file should be rejected."""

    text_file = tmp_path / "document.txt"
    text_file.write_text("Hello")

    with pytest.raises(PDFIngestionError, match="Expected a PDF"):
        ingest_pdf(
            path=text_file,
            document_id="DOC-TEXT",
        )
