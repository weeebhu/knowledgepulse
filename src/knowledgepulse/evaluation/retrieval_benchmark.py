from collections.abc import Callable
from dataclasses import dataclass

from knowledgepulse.evaluation.retrieval_metrics import (
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from knowledgepulse.models.benchmark import BenchmarkQuery


@dataclass(frozen=True)
class RetrievalQueryResult:
    """Evaluation metrics for a single benchmark query."""

    query_id: str
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: set[str]
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


@dataclass(frozen=True)
class RetrievalBenchmarkResult:
    """Aggregated retrieval benchmark metrics."""

    query_count: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    query_results: list[RetrievalQueryResult]


def run_retrieval_benchmark(
    queries: list[BenchmarkQuery],
    retriever: Callable[[str, int], list[str]],
    top_k: int,
) -> RetrievalBenchmarkResult:
    """Run a validated, cutoff-consistent retrieval evaluation."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Benchmark query IDs must be unique")

    if not queries:
        return RetrievalBenchmarkResult(
            query_count=0,
            recall_at_k=0.0,
            precision_at_k=0.0,
            mrr=0.0,
            ndcg_at_k=0.0,
            query_results=[],
        )

    query_results: list[RetrievalQueryResult] = []

    for benchmark_query in queries:
        retrieved_chunk_ids = retriever(benchmark_query.query, top_k)
        if any(not chunk_id.strip() for chunk_id in retrieved_chunk_ids):
            raise ValueError(
                f"Retriever returned an empty chunk ID for query "
                f"'{benchmark_query.query_id}'"
            )
        if len(retrieved_chunk_ids) != len(set(retrieved_chunk_ids)):
            raise ValueError(
                f"Retriever returned duplicate chunk IDs for query "
                f"'{benchmark_query.query_id}'"
            )

        # Every metric, including reciprocal rank, must use the same cutoff.
        retrieved_chunk_ids = retrieved_chunk_ids[:top_k]
        relevant_chunk_ids = set(benchmark_query.relevant_chunk_ids)

        recall_score = recall_at_k(relevant_chunk_ids, retrieved_chunk_ids, top_k)
        precision_score = precision_at_k(relevant_chunk_ids, retrieved_chunk_ids, top_k)
        reciprocal_rank_score = reciprocal_rank(relevant_chunk_ids, retrieved_chunk_ids)
        ndcg_score = ndcg_at_k(relevant_chunk_ids, retrieved_chunk_ids, top_k)

        query_results.append(
            RetrievalQueryResult(
                query_id=benchmark_query.query_id,
                retrieved_chunk_ids=retrieved_chunk_ids,
                relevant_chunk_ids=relevant_chunk_ids,
                recall_at_k=recall_score,
                precision_at_k=precision_score,
                reciprocal_rank=reciprocal_rank_score,
                ndcg_at_k=ndcg_score,
            )
        )

    result_count = len(query_results)
    return RetrievalBenchmarkResult(
        query_count=result_count,
        recall_at_k=sum(result.recall_at_k for result in query_results) / result_count,
        precision_at_k=sum(result.precision_at_k for result in query_results) / result_count,
        mrr=sum(result.reciprocal_rank for result in query_results) / result_count,
        ndcg_at_k=sum(result.ndcg_at_k for result in query_results) / result_count,
        query_results=query_results,
    )
