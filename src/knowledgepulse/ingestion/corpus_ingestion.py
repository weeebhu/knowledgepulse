from pathlib import Path

from knowledgepulse.ingestion.pdf import ingest_pdf
from knowledgepulse.ingestion.text import ingest_text
from knowledgepulse.models.corpus import DocumentCorpus


def ingest_directory(directory: Path) -> DocumentCorpus:
    """Ingest supported PDF and UTF-8 text files in deterministic order."""
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    source_paths = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".pdf", ".txt"}
    )
    documents = []

    for source_path in source_paths:
        document_id = source_path.stem.split("_")[0]
        if source_path.suffix.lower() == ".pdf":
            document = ingest_pdf(source_path, document_id)
        else:
            document = ingest_text(source_path, document_id)
        documents.append(document)

    return DocumentCorpus(documents=documents)
