"""`docpipe-surya process ...`, `doctor`, `evaluate`, `judge`."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .config import ATTRIBUTION, Options, parse_pages


def _utf8_stdio() -> None:
    """The Windows console/pipe code page cannot print Devanagari; without this, print() raises UnicodeEncodeError."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


class _Version(argparse.Action):
    def __init__(self, option_strings, dest, **kwargs):
        super().__init__(option_strings, dest, nargs=0, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        print(f"docpipe-surya {__version__}\n{ATTRIBUTION}")
        parser.exit()


def _pages(value: str) -> str:
    try:
        parse_pages(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return value


def _dpi(value: str) -> int:
    try:
        dpi = int(value)
    except ValueError:
        dpi = 0
    if not 50 <= dpi <= 600:
        raise argparse.ArgumentTypeError(f"--dpi must be a whole number from 50 to 600, got {value!r}")
    return dpi


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docpipe-surya", description="Documents in, JSON and Markdown out. Surya reads only the pixels.")
    parser.add_argument("--version", action=_Version, help="print version and the Surya attribution")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        p.add_argument("--model-dir", type=Path, help="folder with surya-2.gguf and surya-2-mmproj.gguf (default: ./models/surya-gguf or $DOCPIPE_MODEL_DIR)")

    proc = sub.add_parser("process", help="convert documents to document.json + document.md")
    proc.add_argument("inputs", nargs="+", type=Path, metavar="path", help="files or folders")
    proc.add_argument("--out", required=True, type=Path, help="output folder; one subfolder per input document")
    proc.add_argument("--dpi", type=_dpi, default=150, help="render resolution for PDF pages sent to OCR, 50-600 (default 150; cost grows steeply with dpi)")
    proc.add_argument("--force-ocr", action="store_true", help="OCR every PDF page, ignoring text layers")
    proc.add_argument("--pages", type=_pages, help="PDF pages / TIFF frames to process, e.g. 1-5,8")
    proc.add_argument("--max-megapixels", type=float, default=100.0, help="reject images (and lower PDF render dpi) beyond this size")
    proc.add_argument("--deterministic", action="store_true", help="identical output files on every run: no timings or RAM figures, greedy decoding, fixed seed")
    common(proc)

    doc = sub.add_parser("doctor", help="check the installation and the model files")
    common(doc)

    ev = sub.add_parser("evaluate", help="CER, WER, token F1, throughput and review flags for a document.json")
    ev.add_argument("document_json", type=Path)
    ev.add_argument("--reference", required=True, type=Path, help="UTF-8 text file with the ground-truth text")

    jd = sub.add_parser("judge", help="LLM-as-judge: Claude grades sampled OCR pages (sends page images to the Anthropic API)")
    jd.add_argument("output_dir", type=Path, help="folder produced by `docpipe-surya process` (contains <name>/document.json)")
    jd.add_argument("--pdfs", type=Path, help="folder with the source PDFs, if their recorded paths no longer exist")
    jd.add_argument("--pages-per-pdf", type=int, default=3, help="OCR pages sampled per PDF, evenly spread (default 3)")
    jd.add_argument("--model", default="claude-sonnet-5")
    return parser


def _options(args: argparse.Namespace) -> Options:
    opts = Options(device=args.device, model_dir=args.model_dir)
    if args.command == "process":
        opts.dpi = args.dpi
        opts.force_ocr = args.force_ocr
        opts.deterministic = args.deterministic
        opts.pages = args.pages
        opts.max_pixels = int(args.max_megapixels * 1_000_000)
    return opts


def _summary(result) -> str:
    if result.status == "error":
        return f"{result.error_type}: {result.message}"
    ocr = sum(1 for u in result.units if u.origin == "ocr")
    n = len(result.units)
    return f"{n} unit{'s' if n != 1 else ''} ({n - ocr} native, {ocr} ocr), {result.timings.get('total_seconds', 0)}s"


def main(argv: list[str] | None = None) -> int:
    _utf8_stdio()
    args = build_parser().parse_args(argv)
    if args.command == "judge":
        from . import judge

        judge.run(args.output_dir, args.pdfs, args.pages_per_pdf, args.model, say=lambda m: print(m, flush=True))
        return 0
    if args.command == "evaluate":
        import json

        from .metrics import evaluate
        from .schema import DocumentResult

        report = evaluate(DocumentResult.read(args.document_json), args.reference.read_text(encoding="utf-8"))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    opts = _options(args)

    if args.command == "doctor":
        from . import doctor

        return doctor.run(opts)

    from .pipeline import process_batch

    started = time.perf_counter()

    def emit(message: str) -> None:
        elapsed = int(time.perf_counter() - started)
        print(f"[{elapsed // 60:02d}:{elapsed % 60:02d}] {message}", file=sys.stderr, flush=True)

    def report(n: int, total: int, result, folder: Path) -> None:
        print(f"{result.status:<8} {result.source}  ->  {folder}  {_summary(result)}", flush=True)

    batch = process_batch(args.inputs, args.out, opts, emit=emit, on_document=report)
    for path in batch.skipped:
        print(f"skipped  {path}  (unsupported file type)", file=sys.stderr)
    counts = {s: sum(1 for r, _ in batch.documents if r.status == s) for s in ("ok", "partial", "error")}
    print(f"{len(batch.documents)} document(s): {counts['ok']} ok, {counts['partial']} partial, {counts['error']} error. Output: {args.out}")
    return 0 if counts["partial"] == counts["error"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
