"""PPTX: text per slide in slide order (read straight from the XML); pictures are OCR'd after their slide's text."""

from __future__ import annotations

import posixpath
import zipfile
from pathlib import Path

from ..errors import CorruptFile
from ..schema import Unit
from .base import Context, md_table, ocr_embedded_image

A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
P = "{http://schemas.openxmlformats.org/presentationml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
REL = "{http://schemas.openxmlformats.org/package/2006/relationships}"
TITLE_TYPES = {"title", "ctrTitle"}


def _parse(z: zipfile.ZipFile, name: str):
    from lxml import etree

    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False, huge_tree=False)
    try:
        return etree.fromstring(z.read(name), parser)
    except (KeyError, etree.XMLSyntaxError) as exc:
        raise CorruptFile(f"PPTX part {name} is missing or invalid: {exc}") from None


def _rels(z: zipfile.ZipFile, part: str) -> dict[str, str]:
    """Relationship id -> full path inside the package, for the relationships of ``part``."""
    directory, base = posixpath.split(part)
    rels_name = posixpath.join(directory, "_rels", base + ".rels")
    if rels_name not in z.namelist():
        return {}
    out = {}
    for rel in _parse(z, rels_name).iter(REL + "Relationship"):
        target = rel.get("Target", "")
        if rel.get("TargetMode") == "External":
            continue
        out[rel.get("Id")] = target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join(directory, target))
    return out


def _paragraph_text(p) -> str:
    out = []
    for node in p.iter():
        if node.tag == A + "t":
            out.append(node.text or "")
        elif node.tag == A + "br":
            out.append("\n")
    return "".join(out).strip()


def _shapes(element, out: list[tuple[str, object]]) -> None:
    for child in element:
        if child.tag == P + "sp":
            ph = child.find(f"{P}nvSpPr/{P}nvPr/{P}ph")
            title = ph is not None and ph.get("type") in TITLE_TYPES
            for p in child.findall(f"{P}txBody/{A}p"):
                text = _paragraph_text(p)
                if text:
                    out.append(("text", ("## " if title else "") + text))
        elif child.tag == P + "graphicFrame":
            rows = [[" ".join(_paragraph_text(p) for p in tc.iter(A + "p")).strip() for tc in tr.findall(A + "tc")] for tr in child.iter(A + "tr")]
            if any(any(r) for r in rows):
                out.append(("table", rows))
        elif child.tag == P + "pic":
            blip = child.find(f"{P}blipFill/{A}blip")
            if blip is not None and blip.get(R + "embed"):
                out.append(("image", blip.get(R + "embed")))
        elif child.tag == P + "grpSp":
            _shapes(child, out)


def extract(path: Path, ctx: Context) -> list[Unit]:
    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise CorruptFile(f"not a readable PPTX: {exc}") from None
    units: list[Unit] = []
    with z:
        presentation = "ppt/presentation.xml"
        rels = _rels(z, presentation)
        order = [rels[s.get(R + "id")] for s in _parse(z, presentation).iter(P + "sldId") if s.get(R + "id") in rels]
        for number, slide in enumerate(order, 1):
            ctx.emit(f"slide {number}/{len(order)}")
            slide_rels = _rels(z, slide)
            items: list[tuple[str, object]] = []
            _shapes(_parse(z, slide).find(f"{P}cSld/{P}spTree"), items)
            markdown = [md_table(v) if k == "table" else v for k, v in items if k in ("text", "table")]
            if markdown:
                units.append(Unit(0, "page", "native", "PPTX slide text, read directly", text="\n".join(markdown), markdown="\n\n".join(markdown), page=number))
            for kind, value in items:
                if kind != "image":
                    continue
                target = slide_rels.get(value)
                if target is None or target not in z.namelist():
                    ctx.warnings.append(f"slide {number}: picture {value} is not stored in the file")
                    continue
                unit = ocr_embedded_image(ctx, z.read(target), name=target, reason=f"picture on slide {number} ({target})")
                if unit is not None:
                    unit.page = number
                    units.append(unit)
    return units
