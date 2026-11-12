from collections import Counter
from pathlib import Path

from knowledgepulse.benchmarks.dataset import load_benchmark_queries
from knowledgepulse.ingestion.chunking import chunk_document
from knowledgepulse.ingestion.corpus_ingestion import ingest_directory

GOLDEN_DIR = Path("datasets/golden")
ORIGINAL_DATASET = GOLDEN_DIR / "retrieval" / "queries.json"
EXPANDED_DATASET = GOLDEN_DIR / "retrieval" / "queries_expanded.json"


def test_original_golden_dataset_remains_unchanged() -> None:
    queries = load_benchmark_queries(ORIGINAL_DATASET)

    assert len(queries) == 12
    assert len({query.query_id for query in queries}) == 12
    assert all(query.category == "paraphrase" for query in queries)
    assert all(query.is_answerable for query in queries)


def test_expanded_dataset_contains_unique_queries_and_category_coverage() -> None:
    queries = load_benchmark_queries(EXPANDED_DATASET)

    assert len(queries) == 38
    assert len({query.query_id for query in queries}) == 38

    category_counts = Counter(query.category for query in queries)
    assert category_counts == {
        "temporal": 7,
        "multi_document": 10,
        "exact_match": 4,
        "paraphrase": 13,
        "unanswerable": 4,
    }


def test_expanded_dataset_uses_chunk_ids_from_the_ingested_corpus() -> None:
    queries = load_benchmark_queries(EXPANDED_DATASET)
    corpus = ingest_directory(GOLDEN_DIR)
    valid_chunk_ids = {
        chunk.chunk_id
        for document in corpus.documents
        for chunk in chunk_document(document, max_chunk_size=500)
    }

    for query in queries:
        assert set(query.relevant_chunk_ids) <= valid_chunk_ids
        if query.is_answerable:
            assert query.relevant_chunk_ids
        else:
            assert query.category == "unanswerable"
            assert query.relevant_chunk_ids == []


def test_expanded_dataset_answerability_is_consistent() -> None:
    queries = load_benchmark_queries(EXPANDED_DATASET)

    assert sum(query.is_answerable for query in queries) == 34
    assert sum(not query.is_answerable for query in queries) == 4
