"""document.json and document.md, always UTF-8."""

from __future__ import annotations

from pathlib import Path

from .schema import DocumentResult, Unit


def _comment(text: str) -> str:
    return text.replace("--", "- -").replace(">", "&gt;")


def _label(unit: Unit) -> str:
    bits = [f"unit {unit.index}", unit.kind]
    if unit.page is not None:
        bits.append(f"page {unit.page}")
    if unit.name:
        bits.append(unit.name)
    bits.append(unit.origin)
    return " | ".join(bits)


def render_markdown(result: DocumentResult) -> str:
    if result.status == "error":
        return f"> **Processing failed** ({result.error_type}): {result.message}\n"
    parts: list[str] = []
    marked = len(result.units) > 1
    for unit in result.units:
        if marked:
            parts.append(f"<!-- {_comment(_label(unit))} -->")
        if unit.status == "error":
            parts.append(f"> **Unit {unit.index} failed** ({unit.error_type}): {unit.message}")
        elif unit.markdown.strip():
            parts.append(unit.markdown.rstrip())
    return "\n\n".join(parts) + "\n" if parts else ""


def write_document(result: DocumentResult, folder: Path) -> Path:
    """Write ``folder/document.json`` and ``folder/document.md``. Returns the folder."""
    folder.mkdir(parents=True, exist_ok=True)

    markdown = render_markdown(result)
    try:
        md_bytes = markdown.encode("utf-8")
    except UnicodeEncodeError:  # lone surrogates from a hostile or damaged source
        md_bytes = markdown.encode("utf-8", errors="replace")
        result.warnings.append("document.md contains characters that cannot be encoded as UTF-8; they were replaced")

    try:
        json_bytes = result.to_json().encode("utf-8")
    except UnicodeEncodeError:
        result.warnings.append("document.json uses \\u escapes because the text has characters that cannot be encoded as UTF-8")
        json_bytes = result.to_json(ensure_ascii=True).encode("utf-8")

    (folder / "document.md").write_bytes(md_bytes)
    (folder / "document.json").write_bytes(json_bytes)
    return folder
