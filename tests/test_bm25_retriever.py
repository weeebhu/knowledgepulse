import pytest

from knowledgepulse.models.chunk import TextChunk
from knowledgepulse.retrieval.bm25 import BM25Retriever


def make_chunk(
    chunk_id: str,
    text: str,
    chunk_index: int = 0,
) -> TextChunk:
    return TextChunk(
        chunk_id=chunk_id,
        document_id="DOC-001",
        page_number=1,
        chunk_index=chunk_index,
        text=text,
    )


@pytest.fixture
def chunks() -> list[TextChunk]:
    return [
        make_chunk(
            "chunk-backup",
            "Tier-1 database backups run every six hours.",
            0,
        ),
        make_chunk(
            "chunk-deployment",
            "Production deployment requires release manager approval.",
            1,
        ),
        make_chunk(
            "chunk-retention",
            "Database backups are retained for thirty calendar days.",
            2,
        ),
    ]


def test_retrieves_chunk_with_matching_terms(
    chunks: list[TextChunk],
) -> None:
    retriever = BM25Retriever()
    retriever.index(chunks)

    results = retriever.search("database backups", top_k=2)

    assert results
    assert results[0].chunk_id in {
        "chunk-backup",
        "chunk-retention",
    }


def test_returns_at_most_top_k_results(
    chunks: list[TextChunk],
) -> None:
    retriever = BM25Retriever()
    retriever.index(chunks)

    results = retriever.search("database backups", top_k=1)

    assert len(results) == 1


def test_rejects_non_positive_top_k(
    chunks: list[TextChunk],
) -> None:
    retriever = BM25Retriever()
    retriever.index(chunks)

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        retriever.search("database backups", top_k=0)


def test_empty_index_returns_no_results() -> None:
    retriever = BM25Retriever()

    assert retriever.search("database backups", top_k=5) == []


def test_rejects_duplicate_chunk_ids(
    chunks: list[TextChunk],
) -> None:
    retriever = BM25Retriever()

    duplicate = make_chunk(
        chunks[0].chunk_id,
        "A different version of the same chunk.",
    )

    with pytest.raises(ValueError, match="Duplicate chunk ID"):
        retriever.index([chunks[0], duplicate])


def test_rejects_empty_query(
    chunks: list[TextChunk],
) -> None:
    retriever = BM25Retriever()
    retriever.index(chunks)

    with pytest.raises(ValueError, match="query cannot be empty"):
        retriever.search("   ", top_k=5)


def test_results_are_deterministic_for_equal_scores() -> None:
    retriever = BM25Retriever()
    retriever.index(
        [
            make_chunk("chunk-b", "backup policy", 1),
            make_chunk("chunk-a", "backup policy", 0),
        ]
    )

    results = retriever.search("backup policy", top_k=2)

    assert [result.chunk_id for result in results] == [
        "chunk-b",
        "chunk-a",
    ]


def test_indexing_replaces_previous_index(
    chunks: list[TextChunk],
) -> None:
    retriever = BM25Retriever()
    retriever.index(chunks)

    retriever.index([make_chunk("chunk-new", "Disaster recovery snapshots.")])

    results = retriever.search("disaster recovery", top_k=5)

    assert [result.chunk_id for result in results] == ["chunk-new"]
