from knowledgepulse.models.document import IngestedDocument


class DocumentCorpus:
    """A collection of ingested documents."""

    def __init__(self, documents: list[IngestedDocument]) -> None:
        document_ids = [document.document_id for document in documents]

        if len(document_ids) != len(set(document_ids)):
            raise ValueError("Duplicate document ID found")

        self.documents = documents

    @property
    def document_count(self) -> int:
        """Return the number of documents in the corpus."""
        return len(self.documents)

    @property
    def page_count(self) -> int:
        """Return the total number of pages across all documents."""
        return sum(len(document.pages) for document in self.documents)

    @property
    def document_ids(self) -> list[str]:
        """Return all document IDs in the corpus."""
        return [document.document_id for document in self.documents]
