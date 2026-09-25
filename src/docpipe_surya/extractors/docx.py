"""DOCX: paragraphs and tables in document order; each embedded image is OCR'd right where it appears."""

from __future__ import annotations

from pathlib import Path

from ..errors import CorruptFile
from ..schema import Unit
from .base import Context, md_table, ocr_embedded_image

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
V = "{urn:schemas-microsoft-com:vml}"
MC_FALLBACK = "{http://schemas.openxmlformats.org/markup-compatibility/2006}Fallback"


def _events(element):
    """('text', str) and ('image', rId) under one element, in document order."""
    for child in element:
        tag = child.tag
        if tag in (MC_FALLBACK, W + "pPr"):  # Fallback repeats the Choice; pPr/tabs are tab stops, not tabs
            continue
        if tag == W + "t":
            yield "text", child.text or ""
        elif tag == W + "tab":
            yield "text", "\t"
        elif tag in (W + "br", W + "cr"):
            yield "text", "\n"
        elif tag == A + "blip":
            if child.get(R + "embed"):
                yield "image", child.get(R + "embed")
            elif child.get(R + "link"):
                yield "linked", child.get(R + "link")
        elif tag == V + "imagedata" and child.get(R + "id"):
            yield "image", child.get(R + "id")
        yield from _events(child)


def _parts(paragraph_element) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    buffer: list[str] = []
    for kind, value in _events(paragraph_element):
        if kind == "text":
            buffer.append(value)
            continue
        if buffer:
            parts.append(("text", "".join(buffer)))
            buffer = []
        parts.append((kind, value))
    if buffer:
        parts.append(("text", "".join(buffer)))
    return parts


def _body_items(parent):
    """('p' | 'tbl', element) in body order, looking inside content controls."""
    for child in parent.iterchildren():
        if child.tag == W + "p":
            yield "p", child
        elif child.tag == W + "tbl":
            yield "tbl", child
        elif child.tag == W + "sdt":
            content = child.find(W + "sdtContent")
            if content is not None:
                yield from _body_items(content)


def _prefix(paragraph) -> str:
    try:
        name = (paragraph.style.name or "").lower() if paragraph.style is not None else ""
    except Exception:
        name = ""
    if name == "title":
        return "# "
    if name.startswith("heading ") and name.split()[-1].isdigit():
        return "#" * min(int(name.split()[-1]), 6) + " "
    if name.startswith("list bullet"):
        return "- "
    if name.startswith("list number"):
        return "1. "
    ppr = paragraph._p.pPr
    return "- " if ppr is not None and ppr.numPr is not None else ""


def _table(table_element) -> tuple[list[list[str]], list[str]]:
    """Cell text and the image ids found inside the table. Nested tables are flattened into their cell."""
    rows: list[list[str]] = []
    images: list[str] = []
    for tr in table_element.findall(W + "tr"):
        row = []
        for tc in tr.findall(W + "tc"):
            lines = []
            for p in tc.iter(W + "p"):
                text = ""
                for kind, value in _events(p):
                    if kind == "text":
                        text += value
                    elif kind == "image":
                        images.append(value)
                if text.strip():
                    lines.append(text.strip())
            row.append("\n".join(lines))
        rows.append(row)
    return rows, images


def extract(path: Path, ctx: Context) -> list[Unit]:
    import docx
    from docx.text.paragraph import Paragraph

    try:
        document = docx.Document(str(path))
    except Exception as exc:  # python-docx surfaces bad zips and bad XML as several unrelated types
        raise CorruptFile(f"not a readable DOCX: {type(exc).__name__}: {exc}") from None

    related = document.part.related_parts
    units: list[Unit] = []
    markdown: list[str] = []
    text: list[str] = []

    def flush() -> None:
        if markdown:
            units.append(
                Unit(0, "section", "native", "DOCX paragraphs and tables, read directly", text="\n".join(text), markdown="\n\n".join(markdown))
            )
            markdown.clear()
            text.clear()

    def image(rid: str) -> None:
        part = related.get(rid)
        if part is None:
            ctx.warnings.append(f"image {rid} is not stored in the file and was skipped")
            return
        flush()
        unit = ocr_embedded_image(ctx, part.blob, name=str(part.partname).lstrip("/"), reason=f"image embedded in DOCX ({part.partname})")
        if unit is not None:
            units.append(unit)

    for kind, element in _body_items(document.element.body):
        if kind == "tbl":
            rows, images = _table(element)
            if any(any(cell for cell in row) for row in rows):
                markdown.append(md_table(rows))
                text.extend("\t".join(cell.replace("\n", " ") for cell in row) for row in rows)
            for rid in images:
                image(rid)
            continue
        prefix = _prefix(Paragraph(element, document._body))
        for part_kind, value in _parts(element):
            if part_kind == "text":
                if value.strip():
                    markdown.append(prefix + value.strip("\n"))
                    text.append(value.strip("\n"))
            elif part_kind == "image":
                image(value)
            else:
                ctx.warnings.append(f"linked (external) image {value} was not fetched")
    flush()
    return units
