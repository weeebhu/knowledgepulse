import pytest

from knowledgepulse.models.chunk import TextChunk


def test_text_chunk_stores_provenance_and_text() -> None:
    """A text chunk should preserve its source and content."""

    chunk = TextChunk(
        chunk_id="DOC-001-P001-C001",
        document_id="DOC-001",
        page_number=1,
        chunk_index=1,
        text="Production deployments require approval.",
    )

    assert chunk.chunk_id == "DOC-001-P001-C001"
    assert chunk.document_id == "DOC-001"
    assert chunk.page_number == 1
    assert chunk.chunk_index == 1
    assert chunk.text == "Production deployments require approval."


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunk_id", ""),
        ("document_id", ""),
        ("text", ""),
    ],
)
def test_text_chunk_rejects_empty_required_fields(
    field: str,
    value: str,
) -> None:
    """A text chunk should reject empty required fields."""

    kwargs = {
        "chunk_id": "DOC-001-P001-C001",
        "document_id": "DOC-001",
        "page_number": 1,
        "chunk_index": 1,
        "text": "Production deployments require approval.",
    }

    kwargs[field] = value

    with pytest.raises(ValueError):
        TextChunk(**kwargs)


@pytest.mark.parametrize(
    ("page_number", "chunk_index"),
    [
        (0, 1),
        (-1, 1),
        (1, -1),
    ],
)
def test_text_chunk_rejects_invalid_positions(
    page_number: int,
    chunk_index: int,
) -> None:
    """A text chunk should reject invalid provenance positions."""

    with pytest.raises(ValueError):
        TextChunk(
            chunk_id="DOC-001-P001-C001",
            document_id="DOC-001",
            page_number=page_number,
            chunk_index=chunk_index,
            text="Production deployments require approval.",
        )
