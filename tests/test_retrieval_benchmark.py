from pathlib import Path

import pytest

from knowledgepulse.benchmarks.dataset import load_benchmark_queries
from knowledgepulse.evaluation.retrieval_benchmark import run_retrieval_benchmark
from knowledgepulse.models.benchmark import BenchmarkQuery

DATASET_PATH = Path("datasets/golden/retrieval/queries.json")
EXPANDED_DATASET_PATH = Path("datasets/golden/retrieval/queries_expanded.json")


def test_run_retrieval_benchmark_returns_metrics() -> None:
    queries = load_benchmark_queries(DATASET_PATH)

    result = run_retrieval_benchmark(
        queries=queries,
        retriever=lambda query, top_k: [],
        top_k=5,
    )

    assert result.query_count == 12
    assert result.answerable_query_count == 12
    assert result.unanswerable_query_count == 0
    assert result.recall_at_k == 0.0
    assert result.precision_at_k == 0.0
    assert result.mrr == 0.0
    assert result.ndcg_at_k == 0.0
    assert result.category_metrics["paraphrase"].query_count == 12
    assert result.category_metrics["paraphrase"].recall_at_k == 0.0


def test_run_retrieval_benchmark_calculates_metrics() -> None:
    queries = load_benchmark_queries(DATASET_PATH)

    def retriever(query: str, top_k: int) -> list[str]:
        if query.startswith("How often must Tier-1"):
            return ["DOC-003-P001-C000", "DOC-003-P001-C001"]
        return []

    result = run_retrieval_benchmark(queries=[queries[0]], retriever=retriever, top_k=2)

    assert result.query_count == 1
    assert result.answerable_query_count == 1
    assert result.unanswerable_query_count == 0
    assert result.recall_at_k == 1.0
    assert result.precision_at_k == 0.5
    assert result.mrr == 1.0
    assert result.ndcg_at_k == 1.0


def test_run_retrieval_benchmark_rejects_non_positive_top_k() -> None:
    queries = load_benchmark_queries(DATASET_PATH)

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        run_retrieval_benchmark(queries=queries, retriever=lambda query, top_k: [], top_k=0)


def test_run_retrieval_benchmark_returns_per_query_results() -> None:
    queries = load_benchmark_queries(DATASET_PATH)
    result = run_retrieval_benchmark(
        queries=[queries[0]],
        retriever=lambda query, top_k: ["DOC-003-P001-C000"],
        top_k=5,
    )

    assert len(result.query_results) == 1
    query_result = result.query_results[0]
    assert query_result.query_id == "Q001"
    assert query_result.category == "paraphrase"
    assert query_result.retrieved_chunk_ids == ["DOC-003-P001-C000"]
    assert query_result.relevant_chunk_ids == {"DOC-003-P001-C000"}
    assert query_result.recall_at_k == 1.0
    assert query_result.precision_at_k == 1.0
    assert query_result.reciprocal_rank == 1.0
    assert query_result.ndcg_at_k == 1.0


def test_run_retrieval_benchmark_applies_top_k_to_reciprocal_rank() -> None:
    queries = load_benchmark_queries(DATASET_PATH)
    result = run_retrieval_benchmark(
        queries=[queries[0]],
        retriever=lambda query, top_k: [
            "DOC-002-P001-C000",
            "DOC-002-P001-C001",
            "DOC-003-P001-C000",
        ],
        top_k=2,
    )

    assert result.query_results[0].retrieved_chunk_ids == [
        "DOC-002-P001-C000",
        "DOC-002-P001-C001",
    ]
    assert result.mrr == 0.0


def test_run_retrieval_benchmark_rejects_duplicate_query_ids() -> None:
    queries = load_benchmark_queries(DATASET_PATH)
    with pytest.raises(ValueError, match="Benchmark query IDs must be unique"):
        run_retrieval_benchmark(
            queries=[queries[0], queries[0]],
            retriever=lambda query, top_k: [],
            top_k=5,
        )


def test_run_retrieval_benchmark_rejects_duplicate_retrieved_ids() -> None:
    queries = load_benchmark_queries(DATASET_PATH)
    with pytest.raises(ValueError, match="duplicate chunk IDs"):
        run_retrieval_benchmark(
            queries=[queries[0]],
            retriever=lambda query, top_k: ["chunk-1", "chunk-1"],
            top_k=5,
        )


def test_run_retrieval_benchmark_rejects_empty_retrieved_ids() -> None:
    queries = load_benchmark_queries(DATASET_PATH)
    with pytest.raises(ValueError, match="empty chunk ID"):
        run_retrieval_benchmark(
            queries=[queries[0]],
            retriever=lambda query, top_k: ["chunk-1", "  "],
            top_k=5,
        )


def test_category_metrics_are_aggregated_independently() -> None:
    queries = load_benchmark_queries(EXPANDED_DATASET_PATH)
    relevant_by_query = {
        query.query_id: query.relevant_chunk_ids[0]
        for query in queries
        if query.is_answerable
    }

    def retriever(query: str, top_k: int) -> list[str]:
        benchmark_query = next(item for item in queries if item.query == query)
        relevant_id = relevant_by_query[benchmark_query.query_id]
        return [relevant_id] if benchmark_query.category == "temporal" else ["wrong-chunk"]

    result = run_retrieval_benchmark(queries, retriever, top_k=5)

    temporal = result.category_metrics["temporal"]
    paraphrase = result.category_metrics["paraphrase"]
    assert temporal.query_count == 5
    assert temporal.recall_at_k == 1.0
    assert temporal.mrr == 1.0
    assert paraphrase.query_count == 10
    assert paraphrase.recall_at_k == 0.0
    assert paraphrase.mrr == 0.0
    assert sum(item.query_count for item in result.category_metrics.values()) == 24


def test_category_metrics_exclude_unanswerable_queries() -> None:
    queries = load_benchmark_queries(EXPANDED_DATASET_PATH)
    result = run_retrieval_benchmark(queries, lambda query, top_k: [], top_k=5)

    assert "unanswerable" not in result.category_metrics
    assert sum(item.query_count for item in result.category_metrics.values()) == 24


def test_unanswerable_queries_are_excluded_from_retrieval_metrics() -> None:
    queries = load_benchmark_queries(EXPANDED_DATASET_PATH)
    calls: list[str] = []

    def retriever(query: str, top_k: int) -> list[str]:
        calls.append(query)
        return []

    result = run_retrieval_benchmark(queries, retriever, top_k=5)

    assert result.query_count == 28
    assert result.answerable_query_count == 24
    assert result.unanswerable_query_count == 4
    assert len(result.query_results) == 24
    assert len(calls) == 24
    assert all(item.relevant_chunk_ids for item in result.query_results)


def test_all_unanswerable_queries_produce_empty_retrieval_metrics() -> None:
    queries = [
        BenchmarkQuery(
            query_id="Q_NO_ANSWER",
            query="A question outside the available corpus",
            relevant_chunk_ids=[],
            category="unanswerable",
            is_answerable=False,
        )
    ]

    def unexpected_retriever(query: str, top_k: int) -> list[str]:
        pytest.fail("Retriever must not run for an unanswerable query")

    result = run_retrieval_benchmark(queries, unexpected_retriever, top_k=5)

    assert result.query_count == 1
    assert result.answerable_query_count == 0
    assert result.unanswerable_query_count == 1
    assert result.query_results == []
    assert result.category_metrics == {}
    assert result.recall_at_k == 0.0
    assert result.precision_at_k == 0.0
    assert result.mrr == 0.0
    assert result.ndcg_at_k == 0.0
