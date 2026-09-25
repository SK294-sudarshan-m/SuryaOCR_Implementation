"""File type by magic bytes and extension -> format name -> extractor."""

from __future__ import annotations

import importlib
import zipfile
from collections.abc import Callable
from pathlib import Path

from .errors import CorruptFile, EmptyFile, PasswordProtected, UnsupportedFormat

# format -> (module in docpipe_surya.extractors, function). Imported on first use so a JSON run never loads PyMuPDF.
EXTRACTORS: dict[str, tuple[str, str]] = {
    "pdf": ("pdf", "extract"),
    "png": ("image", "extract"),
    "jpeg": ("image", "extract"),
    "tiff": ("image", "extract"),
    "bmp": ("image", "extract"),
    "webp": ("image", "extract"),
    "docx": ("docx", "extract"),
    "xlsx": ("xlsx", "extract"),
    "pptx": ("pptx", "extract"),
    "md": ("text", "extract"),
    "doc": ("legacy", "extract"),
    "xls": ("legacy", "extract"),
    "ppt": ("legacy", "extract"),
}

EXTENSIONS = {
    ".pdf": "pdf", ".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".tif": "tiff", ".tiff": "tiff",
    ".bmp": "bmp", ".webp": "webp", ".docx": "docx", ".xlsx": "xlsx", ".pptx": "pptx", ".md": "md", ".markdown": "md",
    ".doc": "doc", ".xls": "xls", ".ppt": "ppt",
}
_OFFICE_ZIP = {"docx": "word/document.xml", "xlsx": "xl/workbook.xml", "pptx": "ppt/presentation.xml"}
_OLE = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_TEXT_FORMATS = {"md"}


def get_extractor(fmt: str) -> Callable:
    module, name = EXTRACTORS[fmt]
    return getattr(importlib.import_module(f"docpipe_surya.extractors.{module}"), name)


def _sniff(head: bytes) -> str | None:
    if b"%PDF-" in head[:1024]:
        return "pdf"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if head.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return "tiff"
    if head[:2] == b"BM" and len(head) > 18 and int.from_bytes(head[14:18], "little") in (12, 40, 52, 56, 64, 108, 124):
        return "bmp"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
        return "zip"
    if head.startswith(_OLE):
        return "ole"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    return None


def _office_kind(path: Path) -> str:
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
    except zipfile.BadZipFile as exc:
        raise CorruptFile(f"not a valid ZIP/Office container: {exc}") from None
    for fmt, member in _OFFICE_ZIP.items():
        if member in names:
            return fmt
    return "zip"


def detect_format(path: Path) -> tuple[str, list[str]]:
    """Return (format, warnings). Raises EmptyFile, UnsupportedFormat, CorruptFile or PasswordProtected."""
    size = path.stat().st_size
    if size == 0:
        raise EmptyFile("file is empty (0 bytes)")
    with open(path, "rb") as fh:
        head = fh.read(4096)
    by_ext = EXTENSIONS.get(path.suffix.lower())
    magic = _sniff(head)
    warnings: list[str] = []

    if magic == "zip":
        magic = _office_kind(path)
        if magic == "zip":
            raise UnsupportedFormat("ZIP archives are not supported; extract them first")
    if magic == "gif":
        raise UnsupportedFormat("GIF images are not supported")
    if magic == "ole":
        if by_ext in ("doc", "xls", "ppt"):
            return by_ext, warnings
        if by_ext in ("docx", "xlsx", "pptx"):
            raise PasswordProtected(
                f"{path.suffix} file is an OLE container, not OOXML: it is password-protected or a legacy binary "
                "file with the wrong extension; no content was read"
            )
        raise UnsupportedFormat("legacy Office/OLE container with an unrecognised extension")

    if magic is not None:  # binary content identified: it wins over the extension
        if by_ext is not None and by_ext != magic:
            warnings.append(f"extension {path.suffix} says {by_ext} but the content is {magic}; treated as {magic}")
        return magic, warnings

    # No binary signature: only text formats are possible.
    if by_ext in _TEXT_FORMATS:
        return by_ext, warnings
    if by_ext is not None:
        raise CorruptFile(f"content is not a valid {by_ext.upper()} (extension {path.suffix})")
    if path.suffix.lower() in (".eml", ".msg", ".zip", ".7z", ".rar", ".tar", ".gz"):
        raise UnsupportedFormat(f"{path.suffix} containers are not supported; extract the files first")
    raise UnsupportedFormat(f"unsupported file type {path.suffix or '(no extension)'!r}")
