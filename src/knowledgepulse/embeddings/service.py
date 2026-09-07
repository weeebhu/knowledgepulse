from knowledgepulse.embeddings.provider import EmbeddingProvider
from knowledgepulse.models.chunk import TextChunk
from knowledgepulse.models.embedding import EmbeddedChunk


def embed_chunks(
    chunks: list[TextChunk],
    provider: EmbeddingProvider,
) -> list[EmbeddedChunk]:
    """Embed chunks using the provided embedding provider."""

    texts = [chunk.text for chunk in chunks]
    vectors = provider.embed_many(texts)

    embedded_chunks = []

    for chunk, vector in zip(chunks, vectors, strict=True):
        embedded_chunks.append(
            EmbeddedChunk(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
                vector=vector,
                model_name=provider.model_name,
            )
        )

    return embedded_chunks
