import math


def recall_at_k(
    relevant_chunk_ids: set[str],
    retrieved_chunk_ids: list[str],
    k: int,
) -> float:
    """Calculate recall among the top-k retrieved chunks."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_chunk_ids:
        return 0.0

    top_k = retrieved_chunk_ids[:k]
    retrieved_relevant = relevant_chunk_ids.intersection(top_k)

    return len(retrieved_relevant) / len(relevant_chunk_ids)


def precision_at_k(
    relevant_chunk_ids: set[str],
    retrieved_chunk_ids: list[str],
    k: int,
) -> float:
    """Calculate precision among the top-k retrieved chunks."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    top_k = retrieved_chunk_ids[:k]

    if not top_k:
        return 0.0

    retrieved_relevant = relevant_chunk_ids.intersection(top_k)

    return len(retrieved_relevant) / len(top_k)


def reciprocal_rank(
    relevant_chunk_ids: set[str],
    retrieved_chunk_ids: list[str],
) -> float:
    """Calculate reciprocal rank of the first relevant result."""

    if not relevant_chunk_ids:
        raise ValueError("relevant_chunk_ids cannot be empty")

    for position, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant_chunk_ids:
            return 1.0 / position

    return 0.0


def mean_reciprocal_rank(
    query_results: list[tuple[set[str], list[str]]],
) -> float:
    """Calculate mean reciprocal rank across multiple queries."""

    if not query_results:
        return 0.0

    reciprocal_ranks = [
        reciprocal_rank(
            relevant_chunk_ids=relevant_chunk_ids,
            retrieved_chunk_ids=retrieved_chunk_ids,
        )
        for relevant_chunk_ids, retrieved_chunk_ids in query_results
    ]

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def ndcg_at_k(
    relevant_chunk_ids: set[str],
    retrieved_chunk_ids: list[str],
    k: int,
) -> float:
    """Calculate normalized discounted cumulative gain among the top-k results."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    if not relevant_chunk_ids:
        return 0.0

    top_k = retrieved_chunk_ids[:k]

    if not top_k:
        return 0.0

    dcg = 0.0

    for position, chunk_id in enumerate(top_k, start=1):
        if chunk_id in relevant_chunk_ids:
            dcg += 1.0 / math.log2(position + 1)

    ideal_k = min(len(relevant_chunk_ids), k)

    ideal_dcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_k + 1))

    if ideal_dcg == 0.0:
        return 0.0

    return dcg / ideal_dcg
