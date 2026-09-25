"""Pieces every extractor shares: the run context, the OCR-unit builder, encoding and Markdown helpers."""

from __future__ import annotations

import codecs
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from ..config import Options
from ..errors import CorruptFile, DocpipeError
from ..model import OcrEngine
from ..render import Workspace, open_image, save_png
from ..schema import Unit


@dataclass
class Context:
    opts: Options
    workspace: Workspace
    engine: OcrEngine
    warnings: list[str] = field(default_factory=list)  # document-level
    emit: Callable[[str], None] = lambda message: None
    ocr_seconds: float = 0.0
    used_ocr: bool = False


@contextmanager
def _heartbeat(emit: Callable[[str], None], label: str, interval: float = 15.0):
    """Print elapsed time every ``interval`` seconds while a slow call runs, so the console never looks frozen."""
    stop, t0 = threading.Event(), time.perf_counter()

    def beat() -> None:
        while not stop.wait(interval):
            emit(f"    {label} ... {time.perf_counter() - t0:.0f}s elapsed")

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()


def error_unit(exc: Exception, kind: str, origin: str, reason: str, **fields) -> Unit:
    unit = Unit(index=0, kind=kind, origin=origin, reason=reason, status="error", **fields)
    if isinstance(exc, DocpipeError):
        unit.error_type, unit.message = exc.error_type, exc.message
    else:
        unit.error_type, unit.message = type(exc).__name__, str(exc)
    return unit


def ocr_unit(
    ctx: Context,
    png_path: Path,
    *,
    kind: str,
    reason: str,
    render_dpi: float | None,
    warnings: list[str] | None = None,
    **fields,
) -> Unit:
    """Run Surya on one PNG. A failure becomes an error unit; it never propagates."""
    unit = Unit(index=0, kind=kind, origin="ocr", reason=reason, render_dpi=render_dpi, warnings=list(warnings or []), **fields)
    ctx.used_ocr = True
    started = time.perf_counter()
    try:
        with _heartbeat(ctx.emit, "OCR running"):
            page = ctx.engine.recognize(str(png_path))
    except Exception as exc:  # per-unit isolation: a bad page must not stop the document or the batch
        unit.status = "error"
        unit.error_type = exc.error_type if isinstance(exc, DocpipeError) else type(exc).__name__
        unit.message = exc.message if isinstance(exc, DocpipeError) else str(exc)
    else:
        unit.width_px, unit.height_px = page.width, page.height
        unit.blocks = page.blocks
        unit.markdown = page.markdown
        unit.text = "\n\n".join(b["text"] for b in sorted(page.blocks, key=lambda b: b["order"]) if b.get("text"))
        unit.warnings += page.warnings
        if not unit.text.strip():
            unit.warnings.append("OCR found no text")
    unit.seconds = round(time.perf_counter() - started, 2)
    ctx.ocr_seconds += unit.seconds
    ctx.emit(f"    OCR {'done' if unit.status == 'ok' else 'FAILED'} in {unit.seconds}s" + (f" ({len(unit.blocks)} blocks)" if unit.blocks else ""))
    return unit


def ocr_embedded_image(ctx: Context, blob: bytes, *, name: str, reason: str) -> Unit | None:
    """OCR an image embedded in Office/HTML content. Returns None when it was skipped as too small."""
    try:
        frame = next(open_image(blob, ctx.opts))
    except DocpipeError as exc:
        return error_unit(exc, "image", "ocr", reason, name=name)
    if frame.error is not None:
        return error_unit(frame.error, "image", "ocr", reason, name=name)
    w, h = frame.image.size
    if min(w, h) < ctx.opts.min_image_px:
        ctx.warnings.append(f"skipped {name}: only {w}x{h} px")
        return None
    png = save_png(frame.image, ctx.workspace)
    try:
        return ocr_unit(ctx, png, kind="image", reason=reason, render_dpi=None, name=name)
    finally:
        png.unlink(missing_ok=True)


def local_image_bytes(ctx: Context, base_dir: Path, src: str) -> bytes | None:
    """Bytes of an image referenced from HTML/Markdown: a data: URI or a file under ``base_dir``. Remote
    references are never fetched (processing makes no network calls)."""
    import base64
    from urllib.parse import unquote, urlparse

    src = src.strip()
    if src.startswith("data:"):
        header, _, payload = src.partition(",")
        try:
            return base64.b64decode(payload) if ";base64" in header else unquote(payload).encode("latin-1")
        except ValueError:
            ctx.warnings.append("an inline data: image could not be decoded")
            return None
    parsed = urlparse(src)
    if parsed.scheme or src.startswith("//"):
        ctx.warnings.append(f"remote image not fetched (no network calls are made): {src[:80]}")
        return None
    target = (base_dir / unquote(parsed.path)).resolve()
    if base_dir.resolve() not in target.parents:
        ctx.warnings.append(f"image outside the document's folder was not read: {src[:80]}")
        return None
    if not target.is_file():
        ctx.warnings.append(f"image file not found: {src[:80]}")
        return None
    return target.read_bytes()


def decode_text(data: bytes) -> tuple[str, list[str]]:
    """Bytes -> str. BOM, then UTF-8, then Windows-1252 with a warning. Line endings become LF."""
    warnings: list[str] = []
    if data.startswith(codecs.BOM_UTF8):
        text = data[3:].decode("utf-8")
    elif data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        text = data.decode("utf-16")
    else:
        if b"\x00" in data[:8192]:
            raise CorruptFile("file contains NUL bytes; it is not a text file")
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            text = data.decode("cp1252", errors="replace")
            warnings.append("not valid UTF-8; decoded as Windows-1252, so non-Latin text may be wrong")
    return text.replace("\r\n", "\n").replace("\r", "\n"), warnings


def md_table(rows: list[list[str]]) -> str:
    """First row is the header. Cells are escaped so Unicode text and pipes survive."""
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    if width == 0:
        return ""

    def cell(value: str) -> str:
        return value.replace("|", "\\|").replace("\n", "<br>")

    padded = [[cell(c) for c in r] + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(padded[0]) + " |", "| " + " | ".join(["---"] * width) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in padded[1:]]
    return "\n".join(lines)
