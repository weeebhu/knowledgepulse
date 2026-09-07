from dataclasses import dataclass


@dataclass
class TextChunk:
    """A chunk of text with source provenance."""

    chunk_id: str
    document_id: str
    page_number: int
    chunk_index: int
    text: str

    def __post_init__(self) -> None:
        """Validate text chunk invariants."""

        if not self.chunk_id:
            raise ValueError("chunk_id cannot be empty")

        if not self.document_id:
            raise ValueError("document_id cannot be empty")

        if not self.text:
            raise ValueError("text cannot be empty")

        if self.page_number < 1:
            raise ValueError("page_number must be greater than 0")

        if self.chunk_index < 0:
            raise ValueError("chunk_index cannot be negative")
