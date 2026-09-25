"""document.json, always UTF-8."""

from __future__ import annotations

from pathlib import Path

from .schema import DocumentResult


def write_document(result: DocumentResult, folder: Path) -> Path:
    """Write ``folder/document.json``. Returns the folder."""
    folder.mkdir(parents=True, exist_ok=True)
    try:
        json_bytes = result.to_json().encode("utf-8")
    except UnicodeEncodeError:
        result.warnings.append("document.json uses \\u escapes because the text has characters that cannot be encoded as UTF-8")
        json_bytes = result.to_json(ensure_ascii=True).encode("utf-8")
    (folder / "document.json").write_bytes(json_bytes)
    return folder
