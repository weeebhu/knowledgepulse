import pytest

from knowledgepulse.retrieval.vector_store import VectorStore


def test_vector_store_cannot_be_instantiated_directly() -> None:
    """VectorStore should define an abstract retrieval contract."""

    with pytest.raises(TypeError):
        VectorStore()
