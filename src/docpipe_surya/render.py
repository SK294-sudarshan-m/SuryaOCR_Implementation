"""Turning inputs into the PNG files Surya reads: image normalisation, PDF page rendering, temp-dir lifecycle."""

from __future__ import annotations

import io
import math
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from .config import Options, select_pages
from .errors import CorruptFile, DocpipeError, ImageTooLarge


class Workspace:
    """Temporary directory for the normalised pages. Always remove it: ``with Workspace() as ws``."""

    def __init__(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="docpipe-")
        self.path = Path(self._tmp.name)
        self._n = 0

    def new_png(self) -> Path:
        self._n += 1
        return self.path / f"page-{self._n:05d}.png"

    def close(self) -> None:
        self._tmp.cleanup()

    def __enter__(self) -> Workspace:
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def check_pixels(width: int, height: int, opts: Options, what: str = "image") -> None:
    """The guard Surya lacks: it disables Pillow's decompression-bomb check for the whole process."""
    if width * height > opts.max_pixels:
        raise ImageTooLarge(
            f"{what} is {width}x{height} ({width * height / 1e6:.0f} megapixels), over the "
            f"{opts.max_pixels / 1e6:.0f} MP limit; raise it with --max-megapixels or downscale the file"
        )


def _to_rgb(im):
    """RGB with transparency flattened onto white (plain ``convert`` would turn it black)."""
    from PIL import Image

    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        im = im.point(lambda v: v * (1 / 256)).convert("L")
    if im.mode == "P" and "transparency" in im.info:
        im = im.convert("RGBA")
    if im.mode in ("RGBA", "LA"):
        background = Image.new("RGB", im.size, (255, 255, 255))
        background.paste(im.convert("RGBA"), mask=im.convert("RGBA").getchannel("A"))
        return background
    return im.convert("RGB")


def normalize(im):
    """Apply EXIF orientation (Surya does not) and return an RGB image."""
    from PIL import ImageOps

    return _to_rgb(ImageOps.exif_transpose(im))


@dataclass
class ImageFrame:
    index: int  # 0-based frame number
    total: int
    image: object | None  # PIL.Image.Image, RGB, upright; None when this frame could not be read
    error: DocpipeError | None = None  # why, so one bad frame fails only itself


def open_image(source, opts: Options, pages: str | None = None) -> Iterator[ImageFrame]:
    """Yield the frames of an image file or bytes (``pages``: a --pages spec for multi-frame TIFF). Size is
    checked from the header, before any decode."""
    from PIL import Image, UnidentifiedImageError

    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None  # our own check below is the guard, and it honours --max-pixels
    try:
        try:
            im = Image.open(io.BytesIO(source) if isinstance(source, bytes) else source)
        except UnidentifiedImageError as exc:
            raise CorruptFile(f"not a readable image: {exc}") from None
        except (OSError, ValueError, SyntaxError) as exc:
            raise CorruptFile(f"image could not be opened: {exc}") from None

        total = getattr(im, "n_frames", 1)
        for page_number in select_pages(pages if total > 1 else None, total):  # --pages is about paged documents
            index = page_number - 1
            try:
                im.seek(index)
                check_pixels(im.width, im.height, opts, f"image frame {index + 1}" if total > 1 else "image")
                im.load()
                image = normalize(im)
            except ImageTooLarge as exc:
                yield ImageFrame(index, total, None, exc)
                continue
            except (OSError, ValueError, SyntaxError, EOFError) as exc:
                yield ImageFrame(index, total, None, CorruptFile(f"image frame {index + 1} is damaged: {exc}"))
                continue
            yield ImageFrame(index, total, image)
    finally:
        Image.MAX_IMAGE_PIXELS = previous


def save_png(image, workspace: Workspace) -> Path:
    path = workspace.new_png()
    image.save(path, format="PNG")
    return path


@dataclass
class RenderedPage:
    path: Path
    width_px: int
    height_px: int
    dpi: int
    warnings: list[str] = field(default_factory=list)


def render_pdf_page(page, requested_dpi: int, opts: Options, workspace: Workspace) -> RenderedPage:
    """Render at ``requested_dpi``, lowered if the page would exceed the pixel cap."""
    import pymupdf

    rect = page.rect
    dpi = int(requested_dpi)  # PyMuPDF stores dpi as an integer
    warnings: list[str] = []
    pixels = (rect.width / 72 * dpi) * (rect.height / 72 * dpi)
    if pixels > opts.max_pixels:
        dpi = math.floor(dpi * math.sqrt(opts.max_pixels / pixels) * 0.999)
        if dpi < 10:
            raise ImageTooLarge(
                f"PDF page is {rect.width:.0f}x{rect.height:.0f} pt; it cannot be rendered within "
                f"{opts.max_pixels / 1e6:.0f} MP at a usable resolution"
            )
        warnings.append(f"render dpi lowered from {requested_dpi} to {dpi} to stay within {opts.max_pixels / 1e6:.0f} MP")
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csRGB, alpha=False)
    path = workspace.new_png()
    pix.save(path)
    return RenderedPage(path, pix.width, pix.height, dpi, warnings)
