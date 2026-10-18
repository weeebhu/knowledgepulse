import pytest
from pydantic import ValidationError

from knowledgepulse.models.claim import (
    AtomicClaim,
    ClaimRelationship,
    RelationshipType,
)


def test_atomic_claim_preserves_text_and_source_provenance() -> None:
    claim = AtomicClaim(
        claim_id="CLM-001",
        text="Production deployments require approval.",
        document_id="DOC-001",
        chunk_id="DOC-001-P001-C000",
        page_number=1,
    )

    assert claim.claim_id == "CLM-001"
    assert claim.text == "Production deployments require approval."
    assert claim.document_id == "DOC-001"
    assert claim.chunk_id == "DOC-001-P001-C000"
    assert claim.page_number == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("claim_id", ""),
        ("text", ""),
        ("document_id", ""),
        ("chunk_id", ""),
    ],
)
def test_atomic_claim_rejects_empty_required_strings(
    field: str,
    value: str,
) -> None:
    data = {
        "claim_id": "CLM-001",
        "text": "Production deployments require approval.",
        "document_id": "DOC-001",
        "chunk_id": "DOC-001-P001-C000",
        "page_number": 1,
    }
    data[field] = value

    with pytest.raises(ValidationError):
        AtomicClaim(**data)


@pytest.mark.parametrize("page_number", [0, -1])
def test_atomic_claim_rejects_invalid_page_number(
    page_number: int,
) -> None:
    with pytest.raises(ValidationError):
        AtomicClaim(
            claim_id="CLM-001",
            text="Production deployments require approval.",
            document_id="DOC-001",
            chunk_id="DOC-001-P001-C000",
            page_number=page_number,
        )


def test_claim_relationship_preserves_claim_ids_and_type() -> None:
    relationship = ClaimRelationship(
        relationship_id="REL-001",
        source_claim_id="CLM-001",
        target_claim_id="CLM-002",
        relationship_type=RelationshipType.CONTRADICTION,
        evidence="The claims specify incompatible approval requirements.",
        confidence=0.95,
    )

    assert relationship.source_claim_id == "CLM-001"
    assert relationship.target_claim_id == "CLM-002"
    assert relationship.relationship_type == RelationshipType.CONTRADICTION
    assert relationship.confidence == 0.95


@pytest.mark.parametrize(
    "relationship_type",
    [
        RelationshipType.CONTRADICTION,
        RelationshipType.TEMPORAL_SUPERSESSION,
        RelationshipType.COMPATIBILITY,
        RelationshipType.DUPLICATE,
        RelationshipType.AMBIGUOUS,
    ],
)
def test_supported_relationship_types(
    relationship_type: RelationshipType,
) -> None:
    relationship = ClaimRelationship(
        relationship_id="REL-001",
        source_claim_id="CLM-001",
        target_claim_id="CLM-002",
        relationship_type=relationship_type,
    )

    assert relationship.relationship_type == relationship_type


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_claim_relationship_rejects_confidence_out_of_range(
    confidence: float,
) -> None:
    with pytest.raises(ValidationError):
        ClaimRelationship(
            relationship_id="REL-001",
            source_claim_id="CLM-001",
            target_claim_id="CLM-002",
            relationship_type=RelationshipType.CONTRADICTION,
            confidence=confidence,
        )


def test_claim_relationship_rejects_same_claim_as_both_endpoints() -> None:
    with pytest.raises(ValidationError):
        ClaimRelationship(
            relationship_id="REL-001",
            source_claim_id="CLM-001",
            target_claim_id="CLM-001",
            relationship_type=RelationshipType.CONTRADICTION,
        )
