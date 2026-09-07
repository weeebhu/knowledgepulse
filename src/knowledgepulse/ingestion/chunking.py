from knowledgepulse.models.chunk import TextChunk
from knowledgepulse.models.document import IngestedDocument


def chunk_document(
    document: IngestedDocument,
    max_chunk_size: int,
) -> list[TextChunk]:
    """Split a document into provenance-preserving text chunks."""

    if max_chunk_size < 1:
        raise ValueError("max_chunk_size must be greater than 0")

    chunks: list[TextChunk] = []

    for page in document.pages:
        text = page.text.strip()

        if not text:
            continue

        page_chunks = _split_text(
            text=text,
            max_chunk_size=max_chunk_size,
        )

        for chunk_index, chunk_text in enumerate(page_chunks):
            chunk = TextChunk(
                chunk_id=(f"{document.document_id}-P{page.page_number:03d}-C{chunk_index:03d}"),
                document_id=document.document_id,
                page_number=page.page_number,
                chunk_index=chunk_index,
                text=chunk_text,
            )

            chunks.append(chunk)

    return chunks


def _split_text(
    text: str,
    max_chunk_size: int,
) -> list[str]:
    """Split text into chunks within the size limit."""

    words = text.split()
    chunks: list[str] = []
    current_chunk = ""

    for word in words:
        while len(word) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            chunks.append(word[:max_chunk_size])
            word = word[max_chunk_size:]

        if not current_chunk:
            current_chunk = word
            continue

        candidate = f"{current_chunk} {word}"

        if len(candidate) <= max_chunk_size:
            current_chunk = candidate
        else:
            chunks.append(current_chunk)
            current_chunk = word

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
