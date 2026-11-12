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
    """Evaluation metrics for a single answerable benchmark query."""

    query_id: str
    retrieved_chunk_ids: list[str]
    relevant_chunk_ids: set[str]
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


@dataclass(frozen=True)
class RetrievalBenchmarkResult:
    """Retrieval metrics over answerable queries, with dataset coverage counts."""

    query_count: int
    answerable_query_count: int
    unanswerable_query_count: int
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
    """Evaluate ranking quality only where the dataset contains relevant evidence."""

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    query_ids = [query.query_id for query in queries]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Benchmark query IDs must be unique")

    answerable_queries = [query for query in queries if query.is_answerable]
    unanswerable_query_count = len(queries) - len(answerable_queries)

    query_results: list[RetrievalQueryResult] = []

    for benchmark_query in answerable_queries:
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

        # Every ranking metric, including reciprocal rank, uses the same cutoff.
        retrieved_chunk_ids = retrieved_chunk_ids[:top_k]
        relevant_chunk_ids = set(benchmark_query.relevant_chunk_ids)

        query_results.append(
            RetrievalQueryResult(
                query_id=benchmark_query.query_id,
                retrieved_chunk_ids=retrieved_chunk_ids,
                relevant_chunk_ids=relevant_chunk_ids,
                recall_at_k=recall_at_k(relevant_chunk_ids, retrieved_chunk_ids, top_k),
                precision_at_k=precision_at_k(relevant_chunk_ids, retrieved_chunk_ids, top_k),
                reciprocal_rank=reciprocal_rank(relevant_chunk_ids, retrieved_chunk_ids),
                ndcg_at_k=ndcg_at_k(relevant_chunk_ids, retrieved_chunk_ids, top_k),
            )
        )

    answerable_query_count = len(query_results)
    if not answerable_query_count:
        return RetrievalBenchmarkResult(
            query_count=len(queries),
            answerable_query_count=0,
            unanswerable_query_count=unanswerable_query_count,
            recall_at_k=0.0,
            precision_at_k=0.0,
            mrr=0.0,
            ndcg_at_k=0.0,
            query_results=[],
        )

    return RetrievalBenchmarkResult(
        query_count=len(queries),
        answerable_query_count=answerable_query_count,
        unanswerable_query_count=unanswerable_query_count,
        recall_at_k=sum(result.recall_at_k for result in query_results)
        / answerable_query_count,
        precision_at_k=sum(result.precision_at_k for result in query_results)
        / answerable_query_count,
        mrr=sum(result.reciprocal_rank for result in query_results) / answerable_query_count,
        ndcg_at_k=sum(result.ndcg_at_k for result in query_results) / answerable_query_count,
        query_results=query_results,
    )
