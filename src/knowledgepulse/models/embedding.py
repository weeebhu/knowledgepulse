from pydantic import BaseModel, field_validator


class EmbeddedChunk(BaseModel):
    """A text chunk enriched with its vector embedding."""

    chunk_id: str
    document_id: str
    page_number: int
    chunk_index: int
    text: str
    vector: list[float]
    model_name: str

    @field_validator(
        "chunk_id",
        "document_id",
        "text",
        "model_name",
    )
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        """Ensure required string fields are not empty."""
        if not value.strip():
            raise ValueError("Required field cannot be empty")

        return value

    @field_validator("vector")
    @classmethod
    def validate_vector(cls, value: list[float]) -> list[float]:
        """Ensure an embedding contains at least one dimension."""
        if not value:
            raise ValueError("Embedding vector cannot be empty")

        return value
