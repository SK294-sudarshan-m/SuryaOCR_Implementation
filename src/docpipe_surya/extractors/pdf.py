"""PDF: decide per page between the text layer (native) and rendering the page for OCR."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..config import select_pages
from ..errors import CorruptFile, PasswordProtected
from ..quality import Thresholds, judge_text_layer
from ..render import render_pdf_page
from ..schema import Unit
from .base import Context, error_unit, ocr_unit


@dataclass
class Route:
    ocr: bool
    reason: str
    metrics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _union_area(rects) -> float:
    """Area covered by the union of axis-aligned rectangles (x0, y0, x1, y1)."""
    xs = sorted({x for r in rects for x in (r[0], r[2])})
    total = 0.0
    for left, right in zip(xs, xs[1:]):
        spans = sorted((r[1], r[3]) for r in rects if r[0] < right and r[2] > left)
        covered, end = 0.0, float("-inf")
        for lo, hi in spans:
            lo = max(lo, end)
            if hi > lo:
                covered += hi - lo
                end = hi
        total += covered * (right - left)
    return total


def _image_cover(page) -> float:
    import pymupdf

    rect = page.rect
    clipped = []
    for info in page.get_image_info():
        r = pymupdf.Rect(info["bbox"]) & rect
        if not r.is_empty:
            clipped.append((r.x0, r.y0, r.x1, r.y1))
    return _union_area(clipped) / (rect.width * rect.height) if clipped else 0.0


def _text_flags() -> int:
    import pymupdf

    # Expand ligatures (fi, fl) and use U+FFFD, not raw glyph ids, where MuPDF has no Unicode value.
    flags = pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES
    return flags & ~getattr(pymupdf, "TEXT_CID_FOR_UNKNOWN_UNICODE", 0)


def decide(text: str, cover: float, drawing_paths, fonts, force_ocr: bool, t: Thresholds) -> Route:
    """The per-page rule. ``drawing_paths`` is a callable, evaluated only for pages with no text."""
    chars = len("".join(text.split()))
    metrics = {"chars": chars, "image_cover_pct": round(100 * cover)}
    if force_ocr:
        return Route(True, "OCR forced by --force-ocr", metrics)
    if chars == 0:
        if cover >= t.min_image_cover:
            return Route(True, "no text layer", metrics)
        paths = drawing_paths()
        metrics["drawing_paths"] = paths
        if paths >= t.drawing_paths:
            return Route(True, "no text layer (vector graphics only; the text may be outlined)", metrics)
        return Route(False, "blank page (no text, no images)", metrics)
    if chars < t.min_chars:
        if cover >= t.min_image_cover:
            return Route(True, f"text layer too short ({chars} characters) and the page has image content", metrics)
        return Route(False, f"short text layer ({chars} characters), no images", metrics)
    if cover >= t.image_cover and chars < t.sparse_chars:
        return Route(True, f"text layer sparse ({chars} characters) and images cover {cover:.0%} of the page", metrics)

    verdict = judge_text_layer(text, fonts, t)
    metrics.update(verdict.metrics)
    if not verdict.usable:
        return Route(True, verdict.reason, metrics)
    warnings = list(verdict.warnings)
    if cover >= t.image_cover:
        warnings.append(
            f"images cover {cover:.0%} of this page; its text layer may be a scanner's own OCR, "
            "which passed the validity check but was not verified"
        )
    return Route(False, f"text layer valid ({chars} characters)", metrics, warnings)


def _process_page(doc, number: int, ctx: Context) -> Unit:
    page = doc[number - 1]
    rect = page.rect
    blocks = [b for b in page.get_text("blocks", flags=_text_flags()) if b[6] == 0]
    text = "\n".join(b[4].strip("\n") for b in blocks).replace("\xa0", " ")
    fonts = [name for f in page.get_fonts(full=False) for name in (f[3], f[4])]
    route = decide(
        text,
        _image_cover(page),
        lambda: len(page.get_cdrawings()),
        fonts,
        ctx.opts.force_ocr,
        ctx.opts.thresholds,
    )
    common = {"page": number, "page_width_pt": round(rect.width, 1), "page_height_pt": round(rect.height, 1), "text_layer": route.metrics}

    if not route.ocr:
        paragraphs = [b[4].strip("\n").replace("\xa0", " ") for b in blocks if b[4].strip()]
        return Unit(
            0, "page", "native", route.reason,
            text="\n\n".join(paragraphs), markdown="\n\n".join(paragraphs), warnings=route.warnings, **common,
        )

    rendered = render_pdf_page(page, ctx.opts.dpi, ctx.opts, ctx.workspace)
    try:
        return ocr_unit(
            ctx, rendered.path, kind="page", reason=route.reason, render_dpi=rendered.dpi,
            warnings=route.warnings + rendered.warnings, **common,
        )
    finally:
        rendered.path.unlink(missing_ok=True)


def extract(path: Path, ctx: Context) -> list[Unit]:
    import pymupdf

    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # FileDataError, plus RuntimeError/ValueError for truncated or non-PDF input
        raise CorruptFile(f"not a readable PDF: {exc}") from None
    with doc:
        if doc.needs_pass:
            raise PasswordProtected("PDF is encrypted; no text or images were read.")
        if doc.page_count == 0:
            raise CorruptFile("PDF has no pages")
        if doc.is_repaired:
            ctx.warnings.append("PDF structure was damaged and repaired while opening; content may be incomplete")
        pages = select_pages(ctx.opts.pages, doc.page_count)
        if not pages:
            ctx.warnings.append(f"--pages {ctx.opts.pages!r} selects no page of this {doc.page_count}-page PDF")
        units = []
        for number in pages:
            ctx.emit(f"page {number}/{doc.page_count}")
            try:
                units.append(_process_page(doc, number, ctx))
            except Exception as exc:  # per-page isolation; DocpipeError keeps its own type
                units.append(error_unit(exc, "page", "native", "page could not be processed", page=number))
        return units
