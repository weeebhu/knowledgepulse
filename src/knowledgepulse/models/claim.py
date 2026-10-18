from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator


class AtomicClaim(BaseModel):
    """An independently verifiable statement with source provenance."""

    claim_id: str
    text: str
    document_id: str
    chunk_id: str
    page_number: int = Field(ge=1)

    @field_validator("claim_id", "text", "document_id", "chunk_id")
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        """Reject empty or whitespace-only required strings."""

        if not value.strip():
            raise ValueError("Required field cannot be empty")
        return value


class RelationshipType(StrEnum):
    """Supported semantic relationships between two claims."""

    CONTRADICTION = "contradiction"
    TEMPORAL_SUPERSESSION = "temporal_supersession"
    COMPATIBILITY = "compatibility"
    DUPLICATE = "duplicate"
    AMBIGUOUS = "ambiguous"


class ClaimRelationship(BaseModel):
    """A typed relationship between two distinct claims."""

    relationship_id: str
    source_claim_id: str
    target_claim_id: str
    relationship_type: RelationshipType
    evidence: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator(
        "relationship_id",
        "source_claim_id",
        "target_claim_id",
    )
    @classmethod
    def validate_non_empty_string(cls, value: str) -> str:
        """Reject empty or whitespace-only identifiers."""

        if not value.strip():
            raise ValueError("Required field cannot be empty")
        return value

    @field_validator("evidence")
    @classmethod
    def validate_evidence(cls, value: str | None) -> str | None:
        """Reject empty evidence when evidence is supplied."""

        if value is not None and not value.strip():
            raise ValueError("Evidence cannot be empty")
        return value

    @model_validator(mode="after")
    def validate_distinct_claims(self) -> "ClaimRelationship":
        """Ensure a relationship connects two different claims."""

        if self.source_claim_id == self.target_claim_id:
            raise ValueError("A claim relationship requires distinct claims")
        return self
