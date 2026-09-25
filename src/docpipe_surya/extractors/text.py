"""Markdown: read as text, unchanged apart from LF line endings. Local image links are OCR'd."""

from __future__ import annotations

import re
from pathlib import Path

from ..schema import Unit
from .base import Context, decode_text, local_image_bytes, ocr_embedded_image

_IMAGE_LINK = re.compile(r"!\[[^\]]*\]\(\s*<?([^)\s>]+)>?[^)]*\)")


def extract(path: Path, ctx: Context) -> list[Unit]:
    text, warnings = decode_text(path.read_bytes())
    units: list[Unit] = []
    reason = "Markdown file, read directly"
    cursor = 0

    def add_text(chunk: str) -> None:
        if chunk.strip():
            units.append(Unit(0, "section", "native", reason, text=chunk, markdown=chunk, warnings=warnings if not units else []))

    for match in _IMAGE_LINK.finditer(text):
        blob = local_image_bytes(ctx, path.parent, match.group(1))
        if blob is None:
            continue  # leave the link in the text; the reason is in the document warnings
        add_text(text[cursor:match.start()])
        cursor = match.end()
        unit = ocr_embedded_image(ctx, blob, name=match.group(1)[:60], reason=f"image linked from Markdown ({match.group(1)[:60]})")
        if unit is not None:
            units.append(unit)
    add_text(text[cursor:])
    if not units:
        units.append(Unit(0, "section", "native", reason, text=text, markdown=text, warnings=warnings))
    return units
