from pathlib import Path

from knowledgepulse.models.document import DocumentPage, IngestedDocument


class TextIngestionError(Exception):
    """Raised when a plain-text document cannot be ingested."""


def ingest_text(path: Path, document_id: str) -> IngestedDocument:
    """Ingest a UTF-8 text file as a single provenance-preserving page."""
    if not path.exists():
        raise TextIngestionError(f"File does not exist: {path}")

    if path.suffix.lower() != ".txt":
        raise TextIngestionError(f"Expected a TXT file, got: {path.suffix}")

    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise TextIngestionError(f"Failed to read text file: {path}") from exc

    if not text:
        raise TextIngestionError(f"Text file is empty: {path}")

    return IngestedDocument(
        document_id=document_id,
        source_path=path,
        pages=[DocumentPage(page_number=1, text=text)],
    )
