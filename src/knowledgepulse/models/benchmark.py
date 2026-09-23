from pydantic import BaseModel, field_validator


class BenchmarkQuery(BaseModel):
    """A retrieval benchmark query with relevance judgments."""

    query_id: str
    query: str
    relevant_chunk_ids: list[str]

    @field_validator("query_id", "query")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        """Ensure required string fields are not empty."""

        if not value.strip():
            raise ValueError("Required field cannot be empty")

        return value

    @field_validator("relevant_chunk_ids")
    @classmethod
    def validate_relevant_chunk_ids(
        cls,
        value: list[str],
    ) -> list[str]:
        """Ensure relevant chunk IDs are non-empty and unique."""

        if not value:
            raise ValueError("At least one relevant chunk is required")

        if any(not chunk_id.strip() for chunk_id in value):
            raise ValueError("Relevant chunk IDs cannot be empty")

        if len(value) != len(set(value)):
            raise ValueError("Relevant chunk IDs must be unique")

        return value
