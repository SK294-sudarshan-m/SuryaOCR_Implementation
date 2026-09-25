"""Lazy wrapper around Surya v2 (full-page OCR through a local llama.cpp server), plus the checks ``doctor`` reuses.

Nothing here imports torch or surya until the first OCR unit needs them. On CPU, Surya spawns ``llama-server`` bound to
127.0.0.1 and talks to it over loopback HTTP; no other network traffic happens while processing.
"""

from __future__ import annotations

import gc
import importlib.metadata as metadata
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .config import (
    DOWNLOAD_COMMAND,
    GGUF_FILES,
    LLAMA_CPP_RELEASE,
    SURYA_VERSION,
    Options,
    default_llama_server,
    resolve_device,
)
from .errors import ModelUnavailable
from .htmlconv import html_to_markdown, html_to_text


@dataclass
class OcrPage:
    """One page as the engine returns it. ``blocks`` are the output-contract block records."""

    width: int
    height: int
    blocks: list[dict]
    markdown: str
    warnings: list[str] = field(default_factory=list)


class OcrEngine(Protocol):
    def recognize(self, image_path: str) -> OcrPage: ...

    def info(self) -> dict: ...

    def close(self) -> None: ...


@dataclass
class Problem:
    message: str
    fix: str


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def has_nvidia_gpu() -> bool:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return False
    try:
        result = subprocess.run([smi, "-L"], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and "GPU" in result.stdout


def llama_server_version(binary: str) -> str | None:
    try:
        out = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [ln.strip() for ln in (out.stdout + out.stderr).splitlines() if ln.strip().startswith("version")]
    return lines[0] if lines else "unknown"


def model_dir_problems(model_dir: Path) -> list[Problem]:
    problems: list[Problem] = []
    for name in GGUF_FILES:
        if not (model_dir / name).is_file():
            problems.append(
                Problem(
                    f"model file missing: {model_dir / name}",
                    "download it once (not gated): " + DOWNLOAD_COMMAND.format(file=name, model_dir=model_dir),
                )
            )
    return problems


def runtime_problems(device: str, model_dir: Path) -> list[Problem]:
    problems: list[Problem] = []
    installed = package_version("surya-ocr")
    if installed is None:
        problems.append(Problem("surya-ocr is not installed", f"pip install surya-ocr=={SURYA_VERSION}"))
    elif installed != SURYA_VERSION:
        problems.append(Problem(f"surya-ocr {installed} installed, tested with {SURYA_VERSION}", f"pip install surya-ocr=={SURYA_VERSION}"))
    if device == "cpu":
        if default_llama_server() is None:
            problems.append(
                Problem(
                    "llama-server (llama.cpp) not found; Surya needs it for CPU inference",
                    f"download the Windows x64 CPU zip from {LLAMA_CPP_RELEASE}, or set LLAMA_CPP_BINARY",
                )
            )
        problems += model_dir_problems(model_dir)
    elif shutil.which("docker") is None:
        problems.append(Problem("docker not found; Surya's vLLM backend (NVIDIA GPU) runs in Docker", "install Docker and the NVIDIA Container Toolkit, or use --device cpu"))
    return problems


def configure_environment(opts: Options, device: str) -> None:
    """Surya reads its settings from the environment at import time, so this must run first."""
    os.environ["HF_HUB_OFFLINE"] = "1"  # weights are local; processing never touches the network
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if device != "cpu":
        return
    model_dir = opts.resolved_model_dir()
    os.environ["SURYA_INFERENCE_BACKEND"] = "llamacpp"
    os.environ.setdefault("SURYA_INFERENCE_PARALLEL", "1")  # one page at a time; also keeps the KV cache small
    os.environ.setdefault("LLAMA_CPP_BINARY", default_llama_server() or "llama-server")
    os.environ["SURYA_GGUF_LOCAL_MODEL_PATH"] = str(model_dir / GGUF_FILES[0])
    os.environ["SURYA_GGUF_LOCAL_MMPROJ_PATH"] = str(model_dir / GGUF_FILES[1])
    if opts.deterministic:
        extra = os.environ.get("LLAMA_CPP_EXTRA_ARGS", "")
        if "--seed" not in extra:
            os.environ["LLAMA_CPP_EXTRA_ARGS"] = (extra + " --seed 0").strip()


def blocks_from_result(result, width: int, height: int) -> tuple[list[dict], list[str]]:
    """Surya ``PageOCRResult`` -> output-contract blocks (reading order, gap-free ``order``, boxes clipped to the page)."""
    warnings: list[str] = []
    failed = unread = 0
    blocks: list[dict] = []
    for i, blk in enumerate(sorted(result.blocks, key=lambda b: b.reading_order)):
        x0, y0, x1, y1 = blk.bbox
        box = [round(min(max(x0, 0.0), width), 1), round(min(max(y0, 0.0), height), 1), round(min(max(x1, 0.0), width), 1), round(min(max(y1, 0.0), height), 1)]
        html = "" if blk.skipped or blk.error else blk.html
        record = {
            "order": i,
            "label": blk.label,
            "type": blk.raw_label or blk.label,
            "bbox_xyxy": box,
            "conf": round(float(blk.confidence), 4),
            "text": html_to_text(html),
            "html": html,
            "polygon": [[round(x, 1), round(y, 1)] for x, y in blk.polygon],
        }
        if blk.error:
            record["error"] = True
            failed += 1
        elif blk.skipped:
            record["skipped"] = True
        elif not record["text"]:
            unread += 1
        blocks.append(record)
    if failed:
        warnings.append(f"{failed} block(s) failed in the model (kept with empty text)")
    if unread:
        warnings.append(f"{unread} block(s) came back with no text")
    return blocks, warnings


def page_markdown(blocks: list[dict]) -> str:
    return "\n\n".join(md for b in blocks if (md := html_to_markdown(b["html"])))


def _stop_process_windows(pid: int, name: str) -> None:
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)


class SuryaEngine:
    """Surya v2 full-page OCR behind one call. Loads on the first ``recognize``; loading is never retried after it
    failed. ``predictor`` injects a stand-in for ``RecognitionPredictor`` (no server, no weights)."""

    def __init__(self, options: Options, predictor=None) -> None:
        self.opts = options
        self._predictor = predictor
        self._manager = None
        self._load_error: ModelUnavailable | None = None
        self._device: str | None = None
        self.load_seconds = 0.0
        self.status = lambda message: None  # the CLI points this at stderr so a slow start is not silent

    def _load(self) -> None:
        if self._predictor is not None:
            return
        if self._load_error is not None:
            raise self._load_error
        started = time.perf_counter()
        self.status("starting Surya backend (once per run; not part of per-page time)")
        try:
            self._load_unchecked()
        except ModelUnavailable as exc:
            self._load_error = exc
            raise
        finally:
            self.load_seconds += time.perf_counter() - started
        self.status(f"backend ready in {self.load_seconds:.0f}s")

    def _load_unchecked(self) -> None:
        self._device = resolve_device(self.opts.device)
        problems = runtime_problems(self._device, self.opts.resolved_model_dir())
        if problems:
            raise ModelUnavailable("; ".join(f"{p.message} (fix: {p.fix})" for p in problems))
        configure_environment(self.opts, self._device)
        try:
            from surya.inference import SuryaInferenceManager
            from surya.inference.backends import spawn
            from surya.recognition import RecognitionPredictor

            if self.opts.deterministic and "torch" in sys.modules:
                sys.modules["torch"].manual_seed(0)
            if os.name == "nt":
                spawn._stop_process = _stop_process_windows  # Surya's version uses POSIX signals and prints WinError 87
            self._manager = SuryaInferenceManager()
            self._manager.start()
            self._predictor = RecognitionPredictor(self._manager)
        except Exception as exc:  # spawn failures, timeouts and import errors all surface as one clear error
            self._manager = None
            raise ModelUnavailable(f"Surya failed to start: {type(exc).__name__}: {exc}") from exc

    def recognize(self, image_path: str) -> OcrPage:
        from PIL import Image

        self._load()
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        result = self._predictor([image], full_page=True)[0]
        width, height = image.size
        blocks, warnings = blocks_from_result(result, width, height)
        return OcrPage(width, height, blocks, page_markdown(blocks), warnings)

    def info(self) -> dict:
        if self._manager is None:
            return {"engine": "injected test predictor"}
        binary = default_llama_server()
        return {
            "surya-ocr": package_version("surya-ocr"),
            "backend": self._manager.method,
            "device": self._device,
            "llama_server": llama_server_version(binary) if self._manager.method == "llamacpp" and binary else None,
            "model_dir": str(self.opts.resolved_model_dir()),
        }

    def close(self) -> None:
        """Stop the llama-server this run started (an attached, already-running server is left alone)."""
        manager, self._manager, self._predictor = self._manager, None, None
        if manager is not None:
            try:
                handle = manager.backend.handle
                if handle is not None and handle.spawned_by_us:
                    from surya.inference.backends import spawn

                    sentinel = spawn._read_sentinel(manager.method) or {}
                    if sentinel.get("pid"):
                        spawn._stop_process(sentinel["pid"], manager.method)  # patched on Windows in _load_unchecked
                    spawn._delete_sentinel(manager.method)
                manager.stop()
            except Exception:  # cleanup must never mask a result; Surya's atexit hook is the backstop
                pass
        gc.collect()
