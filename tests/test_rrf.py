import pytest

from knowledgepulse.retrieval.rrf import reciprocal_rank_fusion


def test_fuses_two_ranked_lists():
    results = reciprocal_rank_fusion(
        [
            ["A", "B", "C"],
            ["C", "B", "D"],
        ]
    )

    assert results == ["C", "B", "A", "D"]


def test_chunk_present_in_both_lists_gets_fused_score():
    results = reciprocal_rank_fusion(
        [
            ["A", "B"],
            ["B", "A"],
        ]
    )

    assert results[0] in {"A", "B"}
    assert set(results) == {"A", "B"}


def test_handles_chunks_present_in_only_one_list():
    results = reciprocal_rank_fusion(
        [
            ["A"],
            ["B"],
        ]
    )

    assert results == ["A", "B"]


def test_ties_are_resolved_deterministically():
    ranked_lists = [
        ["A", "B"],
        ["B", "A"],
    ]

    first = reciprocal_rank_fusion(ranked_lists)
    second = reciprocal_rank_fusion(ranked_lists)

    assert first == second


def test_empty_ranked_lists_return_empty_results():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


@pytest.mark.parametrize("k", [0, -1])
def test_rejects_non_positive_rrf_constant(k):
    with pytest.raises(ValueError, match="k must be greater than zero"):
        reciprocal_rank_fusion([["A"]], k=k)


def test_rejects_duplicate_chunk_ids_within_a_ranked_list():
    with pytest.raises(ValueError, match="duplicate chunk ID"):
        reciprocal_rank_fusion([["A", "A"]])
