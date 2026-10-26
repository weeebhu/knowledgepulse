import pytest

from knowledgepulse.models.embedding import EmbeddedChunk
from knowledgepulse.retrieval.in_memory_vector_store import InMemoryVectorStore


def make_embedded_chunk(
    chunk_id: str,
    document_id: str,
    text: str = "Document content.",
    vector: list[float] | None = None,
) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        page_number=1,
        chunk_index=0,
        text=text,
        vector=vector or [1.0, 0.0],
        model_name="test-model",
    )


def test_in_memory_vector_store_returns_most_similar_chunks() -> None:
    """The vector store should rank chunks by cosine similarity."""

    store = InMemoryVectorStore()

    chunks = [
        make_embedded_chunk(
            "DOC-001-P001-C000",
            "DOC-001",
            "Production deployment instructions.",
            [1.0, 0.0],
        ),
        make_embedded_chunk(
            "DOC-001-P001-C001",
            "DOC-001",
            "Database backup procedures.",
            [0.0, 1.0],
        ),
        make_embedded_chunk(
            "DOC-001-P001-C002",
            "DOC-001",
            "Deployment approval requirements.",
            [0.9, 0.1],
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
    chunk = make_embedded_chunk("DOC-001-P001-C000", "DOC-001")

    store.add([chunk])
    results = store.search(query_vector=[1.0, 0.0], top_k=10)

    assert len(results) == 1
    assert results[0].chunk.chunk_id == chunk.chunk_id


@pytest.mark.parametrize("top_k", [0, -1])
def test_in_memory_vector_store_rejects_invalid_top_k(top_k: int) -> None:
    """The store should reject non-positive top_k values."""

    store = InMemoryVectorStore()

    with pytest.raises(ValueError):
        store.search(query_vector=[1.0, 0.0], top_k=top_k)


def test_in_memory_vector_store_rejects_dimension_mismatch() -> None:
    """Vectors with different dimensions cannot be compared."""

    store = InMemoryVectorStore()
    store.add(
        [
            make_embedded_chunk(
                "DOC-001-P001-C000",
                "DOC-001",
                vector=[1.0, 0.0, 0.5],
            )
        ]
    )

    with pytest.raises(ValueError):
        store.search(query_vector=[1.0, 0.0], top_k=1)


def test_in_memory_vector_store_preserves_insertion_order_for_equal_scores() -> None:
    """Chunks with equal similarity scores should preserve insertion order."""

    store = InMemoryVectorStore()
    first_chunk = make_embedded_chunk("DOC-001-P001-C000", "DOC-001")
    second_chunk = make_embedded_chunk("DOC-001-P001-C001", "DOC-001")

    store.add([first_chunk, second_chunk])
    results = store.search(query_vector=[1.0, 0.0], top_k=2)

    assert [result.chunk.chunk_id for result in results] == [
        "DOC-001-P001-C000",
        "DOC-001-P001-C001",
    ]


def test_in_memory_vector_store_deletes_chunks_by_document_id() -> None:
    """The vector store should delete only chunks from the requested document."""

    store = InMemoryVectorStore()
    store.add(
        [
            make_embedded_chunk("DOC-001-P001-C000", "DOC-001"),
            make_embedded_chunk(
                "DOC-002-P001-C000",
                "DOC-002",
                vector=[0.0, 1.0],
            ),
        ]
    )

    store.delete_by_document_id("DOC-001")
    results = store.search(query_vector=[0.0, 1.0], top_k=10)

    assert len(results) == 1
    assert results[0].chunk.document_id == "DOC-002"


def test_in_memory_vector_store_returns_similarity_score() -> None:
    """The vector store should return each chunk with its similarity score."""

    store = InMemoryVectorStore()
    chunk = make_embedded_chunk("DOC-001-P001-C000", "DOC-001")

    store.add([chunk])
    results = store.search(query_vector=[1.0, 0.0], top_k=1)

    assert len(results) == 1
    assert results[0].chunk == chunk
    assert results[0].score == 1.0


def test_replace_document_replaces_only_the_target_documents_chunks() -> None:
    store = InMemoryVectorStore()
    store.add(
        [
            make_embedded_chunk("DOC-001-OLD", "DOC-001", "Old version."),
            make_embedded_chunk("DOC-002-KEEP", "DOC-002", "Other document."),
        ]
    )

    store.replace_document(
        "DOC-001",
        [make_embedded_chunk("DOC-001-NEW", "DOC-001", "New version.")],
    )

    results = store.search(query_vector=[1.0, 0.0], top_k=10)

    assert {result.chunk.chunk_id for result in results} == {
        "DOC-001-NEW",
        "DOC-002-KEEP",
    }


def test_replace_document_with_empty_chunks_removes_document() -> None:
    store = InMemoryVectorStore()
    store.add([make_embedded_chunk("DOC-001-OLD", "DOC-001")])

    store.replace_document("DOC-001", [])

    assert store.search(query_vector=[1.0, 0.0], top_k=10) == []


def test_replace_document_validation_failure_preserves_existing_index() -> None:
    store = InMemoryVectorStore()
    original = make_embedded_chunk("DOC-001-OLD", "DOC-001")
    store.add([original])

    with pytest.raises(ValueError, match="target document"):
        store.replace_document(
            "DOC-001",
            [make_embedded_chunk("DOC-002-NEW", "DOC-002")],
        )

    results = store.search(query_vector=[1.0, 0.0], top_k=10)
    assert [result.chunk for result in results] == [original]


def test_add_rejects_duplicate_chunk_ids() -> None:
    store = InMemoryVectorStore()
    chunk = make_embedded_chunk("DOC-001-C000", "DOC-001")

    with pytest.raises(ValueError, match="already exists"):
        store.add([chunk, chunk])
