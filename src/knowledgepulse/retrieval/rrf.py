from collections.abc import Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[str]],
    k: int = 60,
) -> list[str]:
    """Fuse ranked lists of chunk IDs using Reciprocal Rank Fusion."""
    if k <= 0:
        raise ValueError("k must be greater than zero")

    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    next_order = 0

    for ranked_list in ranked_lists:
        if len(ranked_list) != len(set(ranked_list)):
            raise ValueError("duplicate chunk ID in ranked list")

        for rank, chunk_id in enumerate(ranked_list, start=1):
            if chunk_id not in scores:
                scores[chunk_id] = 0.0
                first_seen[chunk_id] = next_order
                next_order += 1

            scores[chunk_id] += 1.0 / (k + rank)

    return sorted(
        scores,
        key=lambda chunk_id: (-scores[chunk_id], first_seen[chunk_id]),
    )
