import pytest

from knowledgepulse.models.embedding import EmbeddedChunk
from knowledgepulse.retrieval.in_memory_vector_store import InMemoryVectorStore


def test_in_memory_vector_store_returns_most_similar_chunks() -> None:
    """The vector store should rank chunks by cosine similarity."""

    store = InMemoryVectorStore()

    chunks = [
        EmbeddedChunk(
            chunk_id="DOC-001-P001-C000",
            document_id="DOC-001",
            page_number=1,
            chunk_index=0,
            text="Production deployment instructions.",
            vector=[1.0, 0.0],
            model_name="test-model",
        ),
        EmbeddedChunk(
            chunk_id="DOC-001-P001-C001",
            document_id="DOC-001",
            page_number=1,
            chunk_index=1,
            text="Database backup procedures.",
            vector=[0.0, 1.0],
            model_name="test-model",
        ),
        EmbeddedChunk(
            chunk_id="DOC-001-P001-C002",
            document_id="DOC-001",
            page_number=1,
            chunk_index=2,
            text="Deployment approval requirements.",
            vector=[0.9, 0.1],
            model_name="test-model",
        ),
    ]

    store.add(chunks)

    results = store.search(
        query_vector=[1.0, 0.0],
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].chunk.chunk_id == "DOC-001-P001-C000"
    assert results[1].chunk.chunk_id == "DOC-001-P001-C002"


def test_in_memory_vector_store_returns_empty_for_empty_store() -> None:
    """Searching an empty store should return no results."""

    store = InMemoryVectorStore()

    results = store.search(
        query_vector=[1.0, 0.0],
        top_k=5,
    )

    assert results == []


def test_in_memory_vector_store_returns_available_results_when_top_k_is_large() -> None:
    """The store should return available results when top_k exceeds storage size."""

    store = InMemoryVectorStore()

    chunk = EmbeddedChunk(
        chunk_id="DOC-001-P001-C000",
        document_id="DOC-001",
        page_number=1,
        chunk_index=0,
        text="Production deployment instructions.",
        vector=[1.0, 0.0],
        model_name="test-model",
    )

    store.add([chunk])

    results = store.search(
        query_vector=[1.0, 0.0],
        top_k=10,
    )

    assert len(results) == 1
    assert results[0].chunk.chunk_id == chunk.chunk_id


@pytest.mark.parametrize(
    "top_k",
    [
        0,
        -1,
    ],
)
def test_in_memory_vector_store_rejects_invalid_top_k(
    top_k: int,
) -> None:
    """The store should reject non-positive top_k values."""

    store = InMemoryVectorStore()

    with pytest.raises(ValueError):
        store.search(
            query_vector=[1.0, 0.0],
            top_k=top_k,
        )


def test_in_memory_vector_store_rejects_dimension_mismatch() -> None:
    """Vectors with different dimensions cannot be compared."""

    store = InMemoryVectorStore()

    chunk = EmbeddedChunk(
        chunk_id="DOC-001-P001-C000",
        document_id="DOC-001",
        page_number=1,
        chunk_index=0,
        text="Production deployment instructions.",
        vector=[1.0, 0.0, 0.5],
        model_name="test-model",
    )

    store.add([chunk])

    with pytest.raises(ValueError):
        store.search(
            query_vector=[1.0, 0.0],
            top_k=1,
        )


def test_in_memory_vector_store_preserves_insertion_order_for_equal_scores() -> None:
    """Chunks with equal similarity scores should preserve insertion order."""

    store = InMemoryVectorStore()

    first_chunk = EmbeddedChunk(
        chunk_id="DOC-001-P001-C000",
        document_id="DOC-001",
        page_number=1,
        chunk_index=0,
        text="First equally relevant chunk.",
        vector=[1.0, 0.0],
        model_name="test-model",
    )

    second_chunk = EmbeddedChunk(
        chunk_id="DOC-001-P001-C001",
        document_id="DOC-001",
        page_number=1,
        chunk_index=1,
        text="Second equally relevant chunk.",
        vector=[1.0, 0.0],
        model_name="test-model",
    )

    store.add([first_chunk, second_chunk])

    results = store.search(
        query_vector=[1.0, 0.0],
        top_k=2,
    )

    assert [result.chunk.chunk_id for result in results] == [
        "DOC-001-P001-C000",
        "DOC-001-P001-C001",
    ]


def test_in_memory_vector_store_deletes_chunks_by_document_id() -> None:
    """The vector store should delete only chunks from the requested document."""

    store = InMemoryVectorStore()

    document_one_chunk = EmbeddedChunk(
        chunk_id="DOC-001-P001-C000",
        document_id="DOC-001",
        page_number=1,
        chunk_index=0,
        text="Document one content.",
        vector=[1.0, 0.0],
        model_name="fake-model",
    )

    document_two_chunk = EmbeddedChunk(
        chunk_id="DOC-002-P001-C000",
        document_id="DOC-002",
        page_number=1,
        chunk_index=0,
        text="Document two content.",
        vector=[0.0, 1.0],
        model_name="fake-model",
    )

    store.add(
        [
            document_one_chunk,
            document_two_chunk,
        ]
    )

    store.delete_by_document_id("DOC-001")

    results = store.search(
        query_vector=[0.0, 1.0],
        top_k=10,
    )

    assert len(results) == 1
    assert results[0].chunk.document_id == "DOC-002"


def test_in_memory_vector_store_returns_similarity_score() -> None:
    """The vector store should return each chunk with its similarity score."""

    store = InMemoryVectorStore()

    chunk = EmbeddedChunk(
        chunk_id="DOC-001-P001-C000",
        document_id="DOC-001",
        page_number=1,
        chunk_index=0,
        text="Production deployments require approval.",
        vector=[1.0, 0.0],
        model_name="fake-model",
    )

    store.add([chunk])

    results = store.search(
        query_vector=[1.0, 0.0],
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].chunk == chunk
    assert results[0].score == 1.0
