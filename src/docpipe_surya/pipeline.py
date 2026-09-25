"""Batch driver: expands inputs, runs one document at a time, isolates failures, writes the outputs."""

from __future__ import annotations

import hashlib
import platform
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__
from .config import Options
from .errors import DocpipeError
from .model import SuryaEngine, OcrEngine, package_version
from .output import write_document
from .render import Workspace
from .router import EXTENSIONS, detect_format, get_extractor
from .schema import DocumentResult

_PAGED = {"pdf", "tiff", "png", "jpeg", "bmp", "webp"}  # single-frame images ignore --pages inside open_image
_FORMAT_LIBS = {
    "pdf": ["PyMuPDF"],
    "docx": ["python-docx", "lxml"],
    "pptx": ["lxml"],
    "xlsx": ["openpyxl"],
    "png": ["Pillow"], "jpeg": ["Pillow"], "tiff": ["Pillow"], "bmp": ["Pillow"], "webp": ["Pillow"],
}


@dataclass
class Job:
    path: Path
    display: str  # the path as the user gave it (or found it under a folder they gave)
    rel_dir: Path  # where under --out the document folder goes


@dataclass
class BatchResult:
    documents: list[tuple[DocumentResult, Path]] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_rss_mb() -> float | None:
    try:
        import psutil

        info = psutil.Process().memory_info()
        own = getattr(info, "peak_wset", info.rss)
        # the llama-server Surya spawns is a child process and holds the model; count its current size too
        children = sum(c.memory_info().rss for c in psutil.Process().children(recursive=True))
        return round((own + children) / 2**20, 1)
    except Exception:
        return None


def _versions(fmt: str | None, used_ocr: bool, engine: OcrEngine) -> dict:
    versions = {"docpipe_surya": __version__, "python": platform.python_version()}
    for lib in _FORMAT_LIBS.get(fmt or "", []):
        versions[lib] = package_version(lib)
    if used_ocr:
        versions["ocr_engine"] = engine.info()
    return versions


def _fail(result: DocumentResult, error_type: str, message: str) -> None:
    result.status = "error"
    result.error_type = error_type
    result.message = message
    result.units = []


def process_document(
    path: Path, display: str, opts: Options, engine: OcrEngine, emit: Callable[[str], None] | None = None
) -> DocumentResult:
    from .extractors.base import Context

    emit = emit or (lambda message: None)
    result = DocumentResult(source=display)
    started = time.perf_counter()
    load_before = getattr(engine, "load_seconds", 0.0)
    ctx = None
    try:
        if not path.is_file():
            raise DocpipeError(f"file not found: {path}", "FileNotFound")
        result.sha256 = sha256_file(path)
        result.format, warnings = detect_format(path)
        result.warnings.extend(warnings)
        if opts.pages and result.format not in _PAGED:
            result.warnings.append(f"--pages applies to PDF pages and multi-frame TIFF frames only; ignored for {result.format}")
        with Workspace() as workspace:
            ctx = Context(opts, workspace, engine, emit=emit)
            units = get_extractor(result.format)(path, ctx)
        for index, unit in enumerate(units):
            unit.index = index
        failed = [u for u in units if u.status == "error"]
        result.warnings.extend(ctx.warnings)
        if units and len(failed) == len(units):
            _fail(result, failed[0].error_type or "Error", failed[0].message or "")
        else:
            result.units = units
            if failed:
                result.status = "partial"
                result.warnings.append(f"{len(failed)} of {len(units)} units failed; see their error_type and message")
            if not units:
                result.warnings.append("no content was extracted")
    except DocpipeError as exc:
        _fail(result, exc.error_type, exc.message)
    except Exception as exc:  # one bad document must never stop the batch
        _fail(result, type(exc).__name__, str(exc))
    result.timings = {
        "total_seconds": round(time.perf_counter() - started, 2),
        "ocr_seconds": round(ctx.ocr_seconds, 2) if ctx else 0.0,
        "model_load_seconds": round(getattr(engine, "load_seconds", 0.0) - load_before, 2),
        "peak_rss_mb": _peak_rss_mb(),
    }
    if opts.deterministic:
        result.timings = {}
        for unit in result.units:
            unit.seconds = None
    result.tool_versions = _versions(result.format, bool(ctx and ctx.used_ocr), engine)
    return result


def collect_inputs(inputs: list[str | Path], out_dir: Path) -> tuple[list[Job], list[str]]:
    jobs: list[Job] = []
    skipped: list[str] = []
    out_resolved = out_dir.resolve()
    for raw in inputs:
        root = Path(raw)
        if not root.is_dir():
            jobs.append(Job(root, str(root), Path(".")))  # a missing file becomes an error record
            continue
        for file in sorted(root.rglob("*")):
            if not file.is_file() or out_resolved in file.resolve().parents:
                continue
            if file.name.startswith(("~$", ".")) or file.suffix.lower() not in EXTENSIONS:
                skipped.append(str(file))
                continue
            jobs.append(Job(file, str(file), file.parent.relative_to(root)))
    return jobs, skipped


def _output_folder(job: Job, out_dir: Path, taken: set[str]) -> tuple[Path, str | None]:
    folder = out_dir / job.rel_dir / job.path.name
    note = None
    candidate, n = folder, 1
    while str(candidate).lower() in taken:
        n += 1
        candidate = folder.with_name(f"{folder.name}-{n}")
    if candidate != folder:
        note = f"another input has the same name; written to {candidate.name} instead"
    taken.add(str(candidate).lower())
    return candidate, note


def process_batch(
    inputs: list[str | Path],
    out_dir: str | Path,
    opts: Options,
    engine: OcrEngine | None = None,
    emit: Callable[[str], None] | None = None,
    on_document: Callable[[int, int, DocumentResult, Path], None] | None = None,
) -> BatchResult:
    out_dir = Path(out_dir)
    emit = emit or (lambda message: None)
    jobs, skipped = collect_inputs(inputs, out_dir)
    own_engine = engine is None
    engine = engine or SuryaEngine(opts)
    if hasattr(engine, "status"):
        engine.status = lambda message: emit(f"  {message}")
    batch = BatchResult(skipped=skipped)
    taken: set[str] = set()
    try:
        for n, job in enumerate(jobs, 1):
            emit(f"[{n}/{len(jobs)}] {job.display}")
            result = process_document(job.path, job.display, opts, engine, lambda message: emit(f"  {message}"))
            folder, note = _output_folder(job, out_dir, taken)
            if note:
                result.warnings.append(note)
            try:
                write_document(result, folder)
            except OSError as exc:  # e.g. disk full: report it, keep going
                _fail(result, "OutputError", f"could not write {folder}: {exc}")
            batch.documents.append((result, folder))
            if on_document:
                on_document(n, len(jobs), result, folder)
    finally:
        if own_engine:
            engine.close()
    return batch
