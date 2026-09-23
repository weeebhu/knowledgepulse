import pytest

from knowledgepulse.evaluation.retrieval_metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_at_k_returns_full_recall_when_all_relevant_chunks_are_retrieved() -> None:
    """Recall should be 1.0 when every relevant chunk is retrieved."""

    relevant_chunk_ids = {
        "DOC-001-P001-C001",
        "DOC-001-P001-C002",
    }
    retrieved_chunk_ids = [
        "DOC-001-P001-C002",
        "DOC-002-P001-C000",
        "DOC-001-P001-C001",
    ]

    assert (
        recall_at_k(
            relevant_chunk_ids=relevant_chunk_ids,
            retrieved_chunk_ids=retrieved_chunk_ids,
            k=3,
        )
        == 1.0
    )


def test_recall_at_k_returns_partial_recall() -> None:
    """Recall should represent the fraction of relevant chunks retrieved."""

    relevant_chunk_ids = {
        "DOC-001-P001-C001",
        "DOC-001-P001-C002",
    }
    retrieved_chunk_ids = [
        "DOC-001-P001-C001",
        "DOC-002-P001-C000",
    ]

    assert (
        recall_at_k(
            relevant_chunk_ids=relevant_chunk_ids,
            retrieved_chunk_ids=retrieved_chunk_ids,
            k=2,
        )
        == 0.5
    )


def test_recall_at_k_only_considers_top_k_results() -> None:
    """Recall should only evaluate the first k retrieved results."""

    relevant_chunk_ids = {
        "DOC-001-P001-C001",
    }
    retrieved_chunk_ids = [
        "DOC-002-P001-C000",
        "DOC-001-P001-C001",
    ]

    assert (
        recall_at_k(
            relevant_chunk_ids=relevant_chunk_ids,
            retrieved_chunk_ids=retrieved_chunk_ids,
            k=1,
        )
        == 0.0
    )


@pytest.mark.parametrize(
    ("relevant_chunk_ids", "retrieved_chunk_ids"),
    [
        (set(), ["DOC-001-P001-C001"]),
        ({"DOC-001-P001-C001"}, []),
    ],
)
def test_recall_at_k_handles_empty_inputs(
    relevant_chunk_ids: set[str],
    retrieved_chunk_ids: list[str],
) -> None:
    """Recall should return zero when there is nothing relevant to retrieve."""

    assert (
        recall_at_k(
            relevant_chunk_ids=relevant_chunk_ids,
            retrieved_chunk_ids=retrieved_chunk_ids,
            k=5,
        )
        == 0.0
    )


@pytest.mark.parametrize("k", [0, -1])
def test_recall_at_k_rejects_non_positive_k(k: int) -> None:
    """Recall should require a positive cutoff."""

    with pytest.raises(ValueError):
        recall_at_k(
            relevant_chunk_ids={"DOC-001-P001-C001"},
            retrieved_chunk_ids=["DOC-001-P001-C001"],
            k=k,
        )


def test_precision_at_k_returns_full_precision_when_all_results_are_relevant() -> None:
    """Precision should be 1.0 when every retrieved result is relevant."""

    relevant_chunk_ids = {
        "DOC-001-P001-C001",
        "DOC-001-P001-C002",
    }
    retrieved_chunk_ids = [
        "DOC-001-P001-C001",
        "DOC-001-P001-C002",
    ]

    assert (
        precision_at_k(
            relevant_chunk_ids=relevant_chunk_ids,
            retrieved_chunk_ids=retrieved_chunk_ids,
            k=2,
        )
        == 1.0
    )


def test_precision_at_k_returns_partial_precision() -> None:
    """Precision should represent the fraction of top-k results that are relevant."""

    relevant_chunk_ids = {
        "DOC-001-P001-C001",
        "DOC-001-P001-C002",
    }
    retrieved_chunk_ids = [
        "DOC-001-P001-C001",
        "DOC-002-P001-C000",
    ]

    assert (
        precision_at_k(
            relevant_chunk_ids=relevant_chunk_ids,
            retrieved_chunk_ids=retrieved_chunk_ids,
            k=2,
        )
        == 0.5
    )


def test_precision_at_k_only_considers_top_k_results() -> None:
    """Precision should only evaluate the first k retrieved results."""

    relevant_chunk_ids = {
        "DOC-001-P001-C001",
    }
    retrieved_chunk_ids = [
        "DOC-002-P001-C000",
        "DOC-001-P001-C001",
    ]

    assert (
        precision_at_k(
            relevant_chunk_ids=relevant_chunk_ids,
            retrieved_chunk_ids=retrieved_chunk_ids,
            k=1,
        )
        == 0.0
    )


@pytest.mark.parametrize(
    ("relevant_chunk_ids", "retrieved_chunk_ids"),
    [
        (set(), ["DOC-001-P001-C001"]),
        ({"DOC-001-P001-C001"}, []),
    ],
)
def test_precision_at_k_handles_empty_inputs(
    relevant_chunk_ids: set[str],
    retrieved_chunk_ids: list[str],
) -> None:
    """Precision should return zero when there are no usable results."""

    assert (
        precision_at_k(
            relevant_chunk_ids=relevant_chunk_ids,
            retrieved_chunk_ids=retrieved_chunk_ids,
            k=5,
        )
        == 0.0
    )


@pytest.mark.parametrize("k", [0, -1])
def test_precision_at_k_rejects_non_positive_k(k: int) -> None:
    """Precision should require a positive cutoff."""

    with pytest.raises(ValueError):
        precision_at_k(
            relevant_chunk_ids={"DOC-001-P001-C001"},
            retrieved_chunk_ids=["DOC-001-P001-C001"],
            k=k,
        )


def test_reciprocal_rank_returns_one_for_first_relevant_result() -> None:
    """Reciprocal rank should be 1 when the first result is relevant."""

    assert (
        reciprocal_rank(
            relevant_chunk_ids={"DOC-001-P001-C001"},
            retrieved_chunk_ids=[
                "DOC-001-P001-C001",
                "DOC-002-P001-C000",
            ],
        )
        == 1.0
    )


def test_reciprocal_rank_returns_inverse_of_first_relevant_position() -> None:
    """Reciprocal rank should use the position of the first relevant result."""

    assert reciprocal_rank(
        relevant_chunk_ids={"DOC-001-P001-C001"},
        retrieved_chunk_ids=[
            "DOC-002-P001-C000",
            "DOC-002-P001-C001",
            "DOC-001-P001-C001",
        ],
    ) == pytest.approx(1 / 3)


def test_reciprocal_rank_ignores_later_relevant_results() -> None:
    """Only the first relevant result should determine reciprocal rank."""

    assert reciprocal_rank(
        relevant_chunk_ids={
            "DOC-001-P001-C001",
            "DOC-001-P001-C002",
        },
        retrieved_chunk_ids=[
            "DOC-002-P001-C000",
            "DOC-001-P001-C002",
            "DOC-001-P001-C001",
        ],
    ) == pytest.approx(1 / 2)


def test_reciprocal_rank_returns_zero_when_no_relevant_result_is_retrieved() -> None:
    """Reciprocal rank should be zero when nothing relevant is retrieved."""

    assert (
        reciprocal_rank(
            relevant_chunk_ids={"DOC-001-P001-C001"},
            retrieved_chunk_ids=[
                "DOC-002-P001-C000",
                "DOC-002-P001-C001",
            ],
        )
        == 0.0
    )


def test_reciprocal_rank_handles_empty_inputs() -> None:
    """Reciprocal rank should return zero for empty inputs."""

    assert (
        reciprocal_rank(
            relevant_chunk_ids={"DOC-001-P001-C001"},
            retrieved_chunk_ids=[],
        )
        == 0.0
    )


def test_reciprocal_rank_rejects_empty_relevance_judgments() -> None:
    """Reciprocal rank should reject a query with no relevant chunks."""

    with pytest.raises(ValueError):
        reciprocal_rank(
            relevant_chunk_ids=set(),
            retrieved_chunk_ids=["DOC-001-P001-C001"],
        )


def test_mean_reciprocal_rank_averages_query_scores() -> None:
    """MRR should average reciprocal rank across queries."""

    query_results = [
        (
            {"DOC-001-P001-C001"},
            ["DOC-001-P001-C001"],
        ),
        (
            {"DOC-002-P001-C001"},
            [
                "DOC-003-P001-C000",
                "DOC-002-P001-C001",
            ],
        ),
        (
            {"DOC-004-P001-C001"},
            ["DOC-003-P001-C000"],
        ),
    ]

    assert mean_reciprocal_rank(query_results) == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_mean_reciprocal_rank_returns_zero_for_empty_queries() -> None:
    """MRR should return zero when there are no evaluated queries."""

    assert mean_reciprocal_rank([]) == 0.0


def test_mean_reciprocal_rank_rejects_empty_relevance_judgment() -> None:
    """MRR should reject queries without relevance judgments."""

    with pytest.raises(ValueError):
        mean_reciprocal_rank(
            [
                (
                    set(),
                    ["DOC-001-P001-C001"],
                )
            ]
        )


def test_ndcg_at_k_returns_one_for_perfect_ranking() -> None:
    relevant_chunk_ids = {"chunk-1", "chunk-2"}
    retrieved_chunk_ids = ["chunk-1", "chunk-2", "chunk-3"]

    assert ndcg_at_k(relevant_chunk_ids, retrieved_chunk_ids, 3) == 1.0


def test_ndcg_at_k_penalizes_relevant_chunks_ranked_lower() -> None:
    relevant_chunk_ids = {"chunk-1", "chunk-2"}
    retrieved_chunk_ids = ["chunk-3", "chunk-1", "chunk-2"]

    score = ndcg_at_k(relevant_chunk_ids, retrieved_chunk_ids, 3)

    assert 0.0 < score < 1.0


def test_ndcg_at_k_only_considers_top_k_results() -> None:
    relevant_chunk_ids = {"chunk-1"}
    retrieved_chunk_ids = ["chunk-2", "chunk-3", "chunk-1"]

    assert ndcg_at_k(relevant_chunk_ids, retrieved_chunk_ids, 2) == 0.0


def test_ndcg_at_k_returns_zero_when_no_relevant_result_is_retrieved() -> None:
    relevant_chunk_ids = {"chunk-1"}
    retrieved_chunk_ids = ["chunk-2", "chunk-3"]

    assert ndcg_at_k(relevant_chunk_ids, retrieved_chunk_ids, 2) == 0.0


def test_ndcg_at_k_returns_zero_for_empty_retrieval() -> None:
    relevant_chunk_ids = {"chunk-1"}

    assert ndcg_at_k(relevant_chunk_ids, [], 5) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_ndcg_at_k_rejects_non_positive_k(k: int) -> None:
    with pytest.raises(ValueError, match="k must be greater than zero"):
        ndcg_at_k({"chunk-1"}, ["chunk-1"], k)
