from pathlib import Path

import pytest

from knowledgepulse.benchmarks.dataset import load_benchmark_queries
from knowledgepulse.ingestion.chunking import chunk_document
from knowledgepulse.ingestion.pdf import ingest_pdf
from knowledgepulse.models.benchmark import BenchmarkQuery


def test_benchmark_query_stores_query_and_relevant_chunks() -> None:
    """A benchmark query should preserve its query and relevance judgments."""

    benchmark_query = BenchmarkQuery(
        query_id="Q001",
        query="How often must Tier-1 production databases generate incremental backups?",
        relevant_chunk_ids=["DOC-003-P001-C000"],
    )

    assert benchmark_query.query_id == "Q001"
    assert (
        benchmark_query.query
        == "How often must Tier-1 production databases generate incremental backups?"
    )
    assert benchmark_query.relevant_chunk_ids == ["DOC-003-P001-C000"]


@pytest.mark.parametrize(
    ("query_id", "query", "relevant_chunk_ids"),
    [
        ("", "A valid query", ["DOC-001-P001-C000"]),
        ("Q001", "", ["DOC-001-P001-C000"]),
        ("Q001", "A valid query", []),
    ],
)
def test_benchmark_query_rejects_invalid_required_fields(
    query_id: str,
    query: str,
    relevant_chunk_ids: list[str],
) -> None:
    """A benchmark query should reject missing required information."""

    with pytest.raises(ValueError):
        BenchmarkQuery(
            query_id=query_id,
            query=query,
            relevant_chunk_ids=relevant_chunk_ids,
        )


def test_benchmark_query_rejects_duplicate_relevant_chunk_ids() -> None:
    """A benchmark query should not list the same chunk more than once."""

    with pytest.raises(ValueError):
        BenchmarkQuery(
            query_id="Q001",
            query="How often are incremental backups generated?",
            relevant_chunk_ids=[
                "DOC-003-P001-C000",
                "DOC-003-P001-C000",
            ],
        )


def test_load_benchmark_queries() -> None:
    """The benchmark dataset should load into validated query models."""

    dataset_path = Path("datasets/golden/retrieval/queries.json")

    queries = load_benchmark_queries(dataset_path)

    assert len(queries) == 12
    assert queries[0].query_id == "Q001"
    assert queries[0].relevant_chunk_ids == ["DOC-003-P001-C000"]


def test_benchmark_query_ids_are_unique() -> None:
    """Benchmark query IDs should be unique."""

    dataset_path = Path("datasets/golden/retrieval/queries.json")

    queries = load_benchmark_queries(dataset_path)

    query_ids = [query.query_id for query in queries]

    assert len(query_ids) == len(set(query_ids))


def test_benchmark_relevance_chunk_ids_have_expected_format() -> None:
    """Relevance judgments should reference valid chunk ID shapes."""

    dataset_path = Path("datasets/golden/retrieval/queries.json")

    queries = load_benchmark_queries(dataset_path)

    for query in queries:
        for chunk_id in query.relevant_chunk_ids:
            parts = chunk_id.split("-")

            assert len(parts) == 4
            assert parts[0] == "DOC"
            assert parts[1].isdigit()
            assert parts[2].startswith("P")
            assert parts[2][1:].isdigit()
            assert parts[3].startswith("C")
            assert parts[3][1:].isdigit()


def test_benchmark_relevance_chunk_ids_exist_in_golden_corpus() -> None:
    """Every judged chunk should exist in the generated golden corpus."""

    dataset_path = Path("datasets/golden/retrieval/queries.json")
    queries = load_benchmark_queries(dataset_path)

    golden_documents = sorted(Path("datasets/golden").glob("*.pdf"))

    chunk_ids = {
        chunk.chunk_id
        for pdf_path in golden_documents
        for chunk in chunk_document(
            document=ingest_pdf(
                pdf_path,
                pdf_path.stem.split("_")[0],
            ),
            max_chunk_size=500,
        )
    }

    judged_chunk_ids = {chunk_id for query in queries for chunk_id in query.relevant_chunk_ids}

    assert judged_chunk_ids <= chunk_ids
