from collections.abc import Callable

from knowledgepulse.retrieval.rrf import reciprocal_rank_fusion


class HybridRetriever:
    """Combine dense and BM25 rankings using Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense: Callable[[str, int], list[str]],
        bm25: Callable[[str, int], list[str]],
        candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")
        self.dense = dense
        self.bm25 = bm25
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k

    def search(self, query: str, top_k: int) -> list[str]:
        """Return fused chunk IDs, retaining the original lightweight API."""
        return [chunk_id for chunk_id, _ in self.search_with_scores(query, top_k)]

    def search_with_scores(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """Return fused IDs and their RRF scores (not raw retriever scores)."""
        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        candidate_k = max(self.candidate_k, top_k)
        rankings = [
            self.dense(query, candidate_k),
            self.bm25(query, candidate_k),
        ]
        ordered_ids = reciprocal_rank_fusion(rankings, k=self.rrf_k)

        scores: dict[str, float] = {}
        for ranking in rankings:
            for rank, chunk_id in enumerate(ranking, start=1):
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)

        return [(chunk_id, scores[chunk_id]) for chunk_id in ordered_ids[:top_k]]
