from knowledgepulse.embeddings.provider import EmbeddingProvider
from knowledgepulse.models.retrieval import RetrievalResult
from knowledgepulse.retrieval.vector_store import VectorStore


def retrieve(
    query: str,
    provider: EmbeddingProvider,
    vector_store: VectorStore,
    top_k: int,
) -> list[RetrievalResult]:
    """Retrieve the most relevant chunks for a query."""

    if not query.strip():
        raise ValueError("query cannot be empty")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")

    query_vector = provider.embed_many([query])[0]

    return vector_store.search(
        query_vector=query_vector,
        top_k=top_k,
    )
