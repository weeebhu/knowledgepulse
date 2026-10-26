from knowledgepulse.embeddings.provider import EmbeddingProvider
from knowledgepulse.embeddings.service import embed_chunks
from knowledgepulse.ingestion.chunking import chunk_document
from knowledgepulse.models.document import IngestedDocument
from knowledgepulse.models.retrieval import RetrievalResult
from knowledgepulse.retrieval.bm25 import BM25Retriever
from knowledgepulse.retrieval.hybrid import HybridRetriever
from knowledgepulse.retrieval.in_memory_vector_store import InMemoryVectorStore


class KnowledgePipeline:
    """End-to-end pipeline for indexing and hybrid-searching documents."""

    def __init__(self, embedding_provider: EmbeddingProvider, max_chunk_size: int) -> None:
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be greater than zero")
        self.embedding_provider = embedding_provider
        self.max_chunk_size = max_chunk_size
        self.vector_store = InMemoryVectorStore()
        self.bm25_retriever = BM25Retriever()

    def index_document(self, document: IngestedDocument) -> None:
        """Prepare both indexes, then publish the replacement document."""
        chunks = chunk_document(document=document, max_chunk_size=self.max_chunk_size)
        embedded_chunks = embed_chunks(chunks=chunks, provider=self.embedding_provider)

        # Build the next lexical index off to the side. If preparation fails,
        # neither currently published index is modified.
        retained_text_chunks = [
            chunk for chunk in self.bm25_retriever.chunks
            if chunk.document_id != document.document_id
        ]
        next_bm25 = BM25Retriever()
        next_bm25.index([*retained_text_chunks, *chunks])

        # Vector replacement validates and builds its next state before swapping.
        self.vector_store.replace_document(document.document_id, embedded_chunks)
        self.bm25_retriever = next_bm25

    def index_documents(self, documents: list[IngestedDocument]) -> None:
        """Index multiple documents in sequence."""
        for document in documents:
            self.index_document(document)

    def search(self, query: str, top_k: int) -> list[RetrievalResult]:
        """Search dense and lexical indexes and fuse their ranked candidates."""
        if not query.strip():
            raise ValueError("query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_vector = self.embedding_provider.embed_many([query])[0]

        def dense_search(_: str, candidate_k: int) -> list[str]:
            results = self.vector_store.search(query_vector, candidate_k)
            return [result.chunk.chunk_id for result in results]

        def lexical_search(search_query: str, candidate_k: int) -> list[str]:
            return [
                chunk.chunk_id
                for chunk in self.bm25_retriever.search(search_query, candidate_k)
            ]

        hybrid = HybridRetriever(dense=dense_search, bm25=lexical_search)
        fused = hybrid.search_with_scores(query, top_k)
        if not fused:
            return []

        resolved = self.vector_store.get_by_ids([chunk_id for chunk_id, _ in fused])
        return [
            RetrievalResult(
                chunk=chunk,
                score=score,
                score_type="reciprocal_rank_fusion",
            )
            for chunk, (_, score) in zip(resolved, fused, strict=True)
        ]
