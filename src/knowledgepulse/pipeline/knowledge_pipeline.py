from knowledgepulse.embeddings.provider import EmbeddingProvider
from knowledgepulse.embeddings.service import embed_chunks
from knowledgepulse.ingestion.chunking import chunk_document
from knowledgepulse.models.document import IngestedDocument
from knowledgepulse.models.retrieval import RetrievalResult
from knowledgepulse.retrieval.in_memory_vector_store import InMemoryVectorStore
from knowledgepulse.retrieval.service import retrieve


class KnowledgePipeline:
    """End-to-end pipeline for indexing and searching documents."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        max_chunk_size: int,
    ) -> None:
        """Initialize the knowledge pipeline."""

        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be greater than zero")

        self.embedding_provider = embedding_provider
        self.max_chunk_size = max_chunk_size
        self.vector_store = InMemoryVectorStore()

    def index_document(
        self,
        document: IngestedDocument,
    ) -> None:
        """Chunk, embed, and index a document."""

        self.vector_store.delete_by_document_id(document.document_id)

        chunks = chunk_document(
            document=document,
            max_chunk_size=self.max_chunk_size,
        )

        embedded_chunks = embed_chunks(
            chunks=chunks,
            provider=self.embedding_provider,
        )

        self.vector_store.add(embedded_chunks)

    def index_documents(
        self,
        documents: list[IngestedDocument],
    ) -> None:
        """Index multiple documents."""

        for document in documents:
            self.index_document(document)

    def search(
        self,
        query: str,
        top_k: int,
    ) -> list[RetrievalResult]:
        """Search indexed documents for relevant chunks."""

        return retrieve(
            query=query,
            provider=self.embedding_provider,
            vector_store=self.vector_store,
            top_k=top_k,
        )
