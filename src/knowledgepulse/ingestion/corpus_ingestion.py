from pathlib import Path

from knowledgepulse.ingestion.pdf import ingest_pdf
from knowledgepulse.models.corpus import DocumentCorpus


def ingest_directory(directory: Path) -> DocumentCorpus:
    """Ingest all PDFs in a directory into a document corpus."""

    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    documents = []

    for pdf_path in sorted(directory.glob("*.pdf")):
        document_id = pdf_path.stem.split("_")[0]

        document = ingest_pdf(
            path=pdf_path,
            document_id=document_id,
        )

        documents.append(document)

    return DocumentCorpus(documents=documents)
