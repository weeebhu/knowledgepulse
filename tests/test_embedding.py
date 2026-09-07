import pytest

from knowledgepulse.models.embedding import EmbeddedChunk


def test_embedded_chunk_preserves_chunk_and_embedding_data() -> None:
    """An embedded chunk should preserve text, provenance, and vector data."""

    embedded_chunk = EmbeddedChunk(
        chunk_id="DOC-001-P001-C000",
        document_id="DOC-001",
        page_number=1,
        chunk_index=0,
        text="Production deployments require approval.",
        vector=[0.1, 0.2, 0.3],
        model_name="test-embedding-model",
    )

    assert embedded_chunk.chunk_id == "DOC-001-P001-C000"
    assert embedded_chunk.document_id == "DOC-001"
    assert embedded_chunk.page_number == 1
    assert embedded_chunk.chunk_index == 0
    assert embedded_chunk.text == "Production deployments require approval."
    assert embedded_chunk.vector == [0.1, 0.2, 0.3]
    assert embedded_chunk.model_name == "test-embedding-model"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("chunk_id", ""),
        ("document_id", ""),
        ("text", ""),
        ("model_name", ""),
    ],
)
def test_embedded_chunk_rejects_empty_required_fields(
    field: str,
    value: str,
) -> None:
    """An embedded chunk should reject empty required fields."""

    kwargs = {
        "chunk_id": "DOC-001-P001-C000",
        "document_id": "DOC-001",
        "page_number": 1,
        "chunk_index": 0,
        "text": "Production deployments require approval.",
        "vector": [0.1, 0.2, 0.3],
        "model_name": "test-embedding-model",
    }

    kwargs[field] = value

    with pytest.raises(ValueError):
        EmbeddedChunk(**kwargs)


def test_embedded_chunk_rejects_empty_vector() -> None:
    """An embedded chunk should require at least one vector dimension."""

    with pytest.raises(ValueError):
        EmbeddedChunk(
            chunk_id="DOC-001-P001-C000",
            document_id="DOC-001",
            page_number=1,
            chunk_index=0,
            text="Production deployments require approval.",
            vector=[],
            model_name="test-embedding-model",
        )
