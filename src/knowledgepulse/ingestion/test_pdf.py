from pathlib import Path

import pymupdf

from knowledgepulse.models.document import DocumentPage, IngestedDocument


class PDFIngestionError(Exception):
    """Raised when a PDF cannot be ingested."""


def ingest_pdf(path: Path, document_id: str) -> IngestedDocument:
    """Extract text from a PDF and return a structured document."""

    if not path.exists():
        raise PDFIngestionError(f"File does not exist: {path}")

    if path.suffix.lower() != ".pdf":
        raise PDFIngestionError(f"Expected a PDF file, got: {path.suffix}")

    try:
        pdf = pymupdf.open(path)
    except Exception as exc:
        raise PDFIngestionError(f"Failed to open PDF: {path}") from exc

    pages = []

    try:
        for index, page in enumerate(pdf):
            text = page.get_text("text").strip()

            pages.append(
                DocumentPage(
                    page_number=index + 1,
                    text=text,
                )
            )
    finally:
        pdf.close()

    return IngestedDocument(
        document_id=document_id,
        source_path=path,
        pages=pages,
    )
