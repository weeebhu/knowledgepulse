from typing import Literal

from pydantic import BaseModel, field_validator, model_validator


BenchmarkQueryCategory = Literal[
    "exact_match",
    "paraphrase",
    "multi_document",
    "temporal",
    "ambiguous",
    "unanswerable",
]


class BenchmarkQuery(BaseModel):
    """A retrieval benchmark query with relevance and answerability judgments."""

    query_id: str
    query: str
    relevant_chunk_ids: list[str]
    category: BenchmarkQueryCategory = "paraphrase"
    is_answerable: bool = True

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
        """Ensure relevance judgments contain unique, non-empty chunk IDs."""

        if any(not chunk_id.strip() for chunk_id in value):
            raise ValueError("Relevant chunk IDs cannot be empty")

        if len(value) != len(set(value)):
            raise ValueError("Relevant chunk IDs must be unique")

        return value

    @model_validator(mode="after")
    def validate_answerability_judgment(self) -> "BenchmarkQuery":
        """Keep answerability labels consistent with relevance judgments."""

        if self.is_answerable and not self.relevant_chunk_ids:
            raise ValueError("Answerable queries require at least one relevant chunk")

        if not self.is_answerable and self.relevant_chunk_ids:
            raise ValueError("Unanswerable queries cannot have relevant chunks")

        if self.category == "unanswerable" and self.is_answerable:
            raise ValueError("Queries in the unanswerable category must not be answerable")

        return self
