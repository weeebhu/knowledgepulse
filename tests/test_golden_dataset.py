from collections import Counter
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
    assert all(query.category == "paraphrase" for query in queries)
    assert all(query.is_answerable for query in queries)


def test_expanded_dataset_contains_unique_queries_and_category_coverage() -> None:
    queries = load_benchmark_queries(EXPANDED_DATASET)

    assert len(queries) == 28
    assert len({query.query_id for query in queries}) == 28

    category_counts = Counter(query.category for query in queries)
    assert category_counts == {
        "temporal": 5,
        "multi_document": 6,
        "exact_match": 3,
        "paraphrase": 10,
        "unanswerable": 4,
    }


def test_expanded_dataset_uses_valid_chunk_ids() -> None:
    queries = load_benchmark_queries(EXPANDED_DATASET)

    for query in queries:
        assert set(query.relevant_chunk_ids) <= VALID_CHUNK_IDS
        if query.is_answerable:
            assert query.relevant_chunk_ids
        else:
            assert query.category == "unanswerable"
            assert query.relevant_chunk_ids == []


def test_expanded_dataset_answerability_is_consistent() -> None:
    queries = load_benchmark_queries(EXPANDED_DATASET)

    assert sum(query.is_answerable for query in queries) == 24
    assert sum(not query.is_answerable for query in queries) == 4
