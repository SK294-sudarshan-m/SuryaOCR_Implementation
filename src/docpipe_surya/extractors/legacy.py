"""Legacy binary Office files (.doc/.xls/.ppt): converted with LibreOffice when it is installed, else unsupported."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from ..errors import CorruptFile, UnsupportedFormat
from ..schema import Unit
from .base import Context

TARGET = {".doc": "docx", ".xls": "xlsx", ".ppt": "pptx"}


def extract(path: Path, ctx: Context) -> list[Unit]:
    from ..router import get_extractor

    soffice = shutil.which("soffice") or shutil.which("soffice.exe")
    suffix = path.suffix.lower()
    if soffice is None:
        raise UnsupportedFormat(
            f"unsupported: {suffix} is a legacy binary format and needs LibreOffice (soffice was not found on PATH); "
            f"convert the file to .{TARGET[suffix]} or install LibreOffice"
        )
    out = ctx.workspace.path / "converted"
    out.mkdir(exist_ok=True)
    try:
        subprocess.run(
            [soffice, "--headless", "--convert-to", TARGET[suffix], "--outdir", str(out), str(path)],
            check=True, capture_output=True, timeout=180,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        raise CorruptFile(f"LibreOffice could not convert {path.name}: {exc}") from None
    converted = out / f"{path.stem}.{TARGET[suffix]}"
    if not converted.is_file():
        raise CorruptFile(f"LibreOffice produced no {TARGET[suffix]} for {path.name}")
    ctx.warnings.append(f"converted {suffix} to {TARGET[suffix]} with LibreOffice before reading")
    return get_extractor(TARGET[suffix])(converted, ctx)
