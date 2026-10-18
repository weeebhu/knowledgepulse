from pathlib import Path

from knowledgepulse.benchmarks.dataset import load_benchmark_queries

GOLDEN_DIR = Path("datasets/golden")
ORIGINAL_DATASET = GOLDEN_DIR / "retrieval" / "queries.json"
EXPANDED_DATASET = GOLDEN_DIR / "retrieval" / "queries_expanded.json"

VALID_CHUNK_IDS = {
    f"DOC-{document_number:03d}-P001-C{chunk_number:03d}"
    for document_number in range(1, 5)
    for chunk_number in range(3)
}


def test_original_golden_dataset_remains_unchanged() -> None:
    queries = load_benchmark_queries(ORIGINAL_DATASET)

    assert len(queries) == 12
    assert len({query.query_id for query in queries}) == 12


def test_expanded_dataset_contains_24_unique_queries() -> None:
    queries = load_benchmark_queries(EXPANDED_DATASET)

    assert len(queries) == 24
    assert len({query.query_id for query in queries}) == 24


def test_expanded_dataset_uses_valid_chunk_ids() -> None:
    queries = load_benchmark_queries(EXPANDED_DATASET)

    for query in queries:
        assert query.relevant_chunk_ids
        assert set(query.relevant_chunk_ids) <= VALID_CHUNK_IDS
