from pathlib import Path

import pytest

from knowledgepulse.ingestion.text import TextIngestionError, ingest_text


def test_ingest_text_extracts_utf8_content(tmp_path: Path) -> None:
    source = tmp_path / "DOC-005_policy.txt"
    source.write_text("Access requires approval.\nMFA is mandatory.", encoding="utf-8")

    document = ingest_text(source, "DOC-005")

    assert document.document_id == "DOC-005"
    assert document.source_path == source
    assert len(document.pages) == 1
    assert document.pages[0].page_number == 1
    assert document.text == "Access requires approval.\nMFA is mandatory."


def test_ingest_text_rejects_missing_file() -> None:
    with pytest.raises(TextIngestionError, match="File does not exist"):
        ingest_text(Path("missing.txt"), "DOC-MISSING")


def test_ingest_text_rejects_non_text_file(tmp_path: Path) -> None:
    source = tmp_path / "policy.pdf"
    source.write_text("not a text source", encoding="utf-8")

    with pytest.raises(TextIngestionError, match="Expected a TXT file"):
        ingest_text(source, "DOC-TEST")


def test_ingest_text_rejects_empty_file(tmp_path: Path) -> None:
    source = tmp_path / "empty.txt"
    source.write_text("  \n", encoding="utf-8")

    with pytest.raises(TextIngestionError, match="Text file is empty"):
        ingest_text(source, "DOC-EMPTY")
