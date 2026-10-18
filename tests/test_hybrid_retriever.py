from knowledgepulse.retrieval.hybrid import HybridRetriever


class FakeDenseRetriever:
    def __init__(self, rankings: dict[str, list[str]]) -> None:
        self.rankings = rankings
        self.calls: list[tuple[str, int]] = []

    def __call__(self, query: str, top_k: int) -> list[str]:
        self.calls.append((query, top_k))
        return self.rankings.get(query, [])[:top_k]


class FakeBM25Retriever:
    def __init__(self, rankings: dict[str, list[str]]) -> None:
        self.rankings = rankings
        self.calls: list[tuple[str, int]] = []

    def __call__(self, query: str, top_k: int) -> list[str]:
        self.calls.append((query, top_k))
        return self.rankings.get(query, [])[:top_k]


def test_fuses_dense_and_bm25_rankings() -> None:
    dense = FakeDenseRetriever({"deployment": ["A", "B", "C"]})
    bm25 = FakeBM25Retriever({"deployment": ["C", "B", "D"]})
    retriever = HybridRetriever(dense=dense, bm25=bm25)

    results = retriever.search("deployment", top_k=3)

    assert results == ["C", "B", "A"]


def test_requests_more_candidates_than_final_top_k() -> None:
    dense = FakeDenseRetriever({"query": ["A", "B", "C", "D", "E"]})
    bm25 = FakeBM25Retriever({"query": ["B", "A", "D", "C", "E"]})
    retriever = HybridRetriever(
        dense=dense,
        bm25=bm25,
        candidate_k=5,
    )

    retriever.search("query", top_k=2)

    assert dense.calls == [("query", 5)]
    assert bm25.calls == [("query", 5)]


def test_returns_empty_list_when_both_retrievers_return_no_results() -> None:
    dense = FakeDenseRetriever({})
    bm25 = FakeBM25Retriever({})
    retriever = HybridRetriever(dense=dense, bm25=bm25)

    assert retriever.search("unknown", top_k=5) == []


def test_rejects_empty_query() -> None:
    dense = FakeDenseRetriever({})
    bm25 = FakeBM25Retriever({})
    retriever = HybridRetriever(dense=dense, bm25=bm25)

    import pytest

    with pytest.raises(ValueError, match="query cannot be empty"):
        retriever.search("  ", top_k=5)


def test_rejects_non_positive_top_k() -> None:
    dense = FakeDenseRetriever({})
    bm25 = FakeBM25Retriever({})
    retriever = HybridRetriever(dense=dense, bm25=bm25)

    import pytest

    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        retriever.search("query", top_k=0)
