from pathlib import Path

import pytest

from knowledgepulse.benchmarks.dataset import load_benchmark_queries
from knowledgepulse.evaluation.retrieval_benchmark import run_retrieval_benchmark

DATASET_PATH = Path("datasets/golden/retrieval/queries.json")


def test_run_retrieval_benchmark_returns_metrics() -> None:
    queries = load_benchmark_queries(DATASET_PATH)

    result = run_retrieval_benchmark(
        queries=queries,
        retriever=lambda query, top_k: [],
        top_k=5,
    )

    assert result.query_count == 12
    assert result.recall_at_k == 0.0
    assert result.precision_at_k == 0.0
    assert result.mrr == 0.0
    assert result.ndcg_at_k == 0.0


def test_run_retrieval_benchmark_calculates_metrics() -> None:
    queries = load_benchmark_queries(DATASET_PATH)

    def retriever(query: str, top_k: int) -> list[str]:
        if query.startswith("How often must Tier-1"):
            return [
                "DOC-003-P001-C000",
                "DOC-003-P001-C001",
            ]

        return []

    result = run_retrieval_benchmark(
        queries=[queries[0]],
        retriever=retriever,
        top_k=2,
    )

    assert result.query_count == 1
    assert result.recall_at_k == 1.0
    assert result.precision_at_k == 0.5
    assert result.mrr == 1.0
    assert result.ndcg_at_k == 1.0


def test_run_retrieval_benchmark_rejects_non_positive_top_k() -> None:
    queries = load_benchmark_queries(DATASET_PATH)

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        run_retrieval_benchmark(
            queries=queries,
            retriever=lambda query, top_k: [],
            top_k=0,
        )


def test_run_retrieval_benchmark_returns_per_query_results() -> None:
    queries = load_benchmark_queries(DATASET_PATH)

    def retriever(query: str, top_k: int) -> list[str]:
        return ["DOC-003-P001-C000"]

    result = run_retrieval_benchmark(
        queries=[queries[0]],
        retriever=retriever,
        top_k=5,
    )

    assert len(result.query_results) == 1

    query_result = result.query_results[0]

    assert query_result.query_id == "Q001"
    assert query_result.retrieved_chunk_ids == ["DOC-003-P001-C000"]
    assert query_result.relevant_chunk_ids == {"DOC-003-P001-C000"}
    assert query_result.recall_at_k == 1.0
    assert query_result.precision_at_k == 1.0
    assert query_result.reciprocal_rank == 1.0
    assert query_result.ndcg_at_k == 1.0
