"""PNG / JPEG / TIFF / BMP / WEBP: normalised, written as a temporary PNG, OCR'd. One unit per frame."""

from __future__ import annotations

from pathlib import Path

from ..render import open_image, save_png
from ..schema import Unit
from .base import Context, error_unit, ocr_unit


def extract(path: Path, ctx: Context) -> list[Unit]:
    units: list[Unit] = []
    for frame in open_image(path, ctx.opts, ctx.opts.pages):
        ctx.emit(f"frame {frame.index + 1}/{frame.total}")
        kind = "page" if frame.total > 1 else "image"
        page = frame.index + 1 if frame.total > 1 else None
        if frame.error is not None:
            units.append(error_unit(frame.error, kind, "ocr", "image frame could not be read", page=page))
            continue
        png = save_png(frame.image, ctx.workspace)
        try:
            units.append(ocr_unit(ctx, png, kind=kind, reason="image file (pixels only, no text layer)", render_dpi=None, page=page))
        finally:
            png.unlink(missing_ok=True)
    if not units:
        ctx.warnings.append(f"--pages {ctx.opts.pages!r} selected no frame of this image")
    return units
