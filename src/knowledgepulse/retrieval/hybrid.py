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
        """Retrieve candidates from both systems and fuse their rankings."""
        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        candidate_k = max(self.candidate_k, top_k)

        dense_results = self.dense(query, candidate_k)
        bm25_results = self.bm25(query, candidate_k)

        fused_results = reciprocal_rank_fusion(
            [dense_results, bm25_results],
            k=self.rrf_k,
        )
        return fused_results[:top_k]
