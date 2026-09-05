from pathlib import Path

from pydantic import BaseModel, Field


class DocumentPage(BaseModel):
    """Represents extracted content from a single document page."""

    page_number: int = Field(..., ge=1)
    text: str


class IngestedDocument(BaseModel):
    """Represents a document after ingestion and text extraction."""

    document_id: str
    source_path: Path
    pages: list[DocumentPage]

    @property
    def text(self) -> str:
        """Return the complete document text."""
        return "\n".join(page.text for page in self.pages)
