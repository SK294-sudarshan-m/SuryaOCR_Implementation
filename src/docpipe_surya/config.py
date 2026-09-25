"""Options, defaults, and the pinned model coordinates."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .quality import Thresholds

ATTRIBUTION = "OCR by Surya (Datalab), modified OpenRAIL-M weights"
LICENCE_NOTE = (
    "Surya code is Apache 2.0. The model weights are under a modified AI Pubs OpenRAIL-M licence: free for research, "
    "personal use and startups under $5M funding/revenue; otherwise a Datalab commercial licence is needed. "
    "Do not offer this tool as a network service to third parties."
)

SURYA_VERSION = "0.22.1"
GGUF_REPO = "datalab-to/surya-ocr-2-gguf"
GGUF_FILES = ("surya-2.gguf", "surya-2-mmproj.gguf")  # model + multimodal projector, about 1.4 GB together
LLAMA_CPP_RELEASE = "https://github.com/ggml-org/llama.cpp/releases (Windows x64 CPU build, unzip into tools\\llama.cpp)"
DOWNLOAD_COMMAND = f"hf download {GGUF_REPO} {{file}} --local-dir {{model_dir}}"


@dataclass
class Options:
    dpi: int = 150  # see progress.txt: prefill cost grows with dpi (about 20 s at 96, 60 s at 150, 140+ s at 200 per page on this CPU)
    force_ocr: bool = False
    pages: str | None = None  # "1-5,8"; applies to PDF pages
    device: str = "auto"
    model_dir: Path | None = None
    max_pixels: int = 100_000_000  # per image, and per rendered PDF page
    deterministic: bool = False  # omit timings/RAM from the output and seed torch, so re-runs give identical files
    min_image_px: int = 32  # embedded images smaller than this on either side are skipped
    thresholds: Thresholds = field(default_factory=Thresholds)

    def resolved_model_dir(self) -> Path:
        return Path(self.model_dir) if self.model_dir else default_model_dir()


def default_model_dir() -> Path:
    """``DOCPIPE_MODEL_DIR``, else ``./models/surya-gguf``, else the copy beside the source tree."""
    env = os.environ.get("DOCPIPE_MODEL_DIR")
    if env:
        return Path(env)
    here = Path.cwd() / "models" / "surya-gguf"
    if here.is_dir():
        return here
    beside_source = Path(__file__).resolve().parents[2] / "models" / "surya-gguf"
    return beside_source if beside_source.is_dir() else here


def default_llama_server() -> str | None:
    """``LLAMA_CPP_BINARY``, else ``tools/llama.cpp/llama-server.exe`` (cwd, then beside the source tree), else PATH."""
    import shutil

    env = os.environ.get("LLAMA_CPP_BINARY")
    if env and Path(env).is_file():
        return env
    for base in (Path.cwd(), Path(__file__).resolve().parents[2]):
        for name in ("llama-server.exe", "llama-server"):
            candidate = base / "tools" / "llama.cpp" / name
            if candidate.is_file():
                return str(candidate)
    return shutil.which("llama-server")


def resolve_device(requested: str) -> str:
    """cpu: llama.cpp backend. cuda: Surya's default (vLLM in Docker) when an NVIDIA GPU is present."""
    if requested in ("cpu", "cuda"):
        return requested
    from .model import has_nvidia_gpu

    return "cuda" if has_nvidia_gpu() else "cpu"


def parse_pages(spec: str) -> list[tuple[int, int | None]]:
    """``"1-3,5,8-"`` -> [(1, 3), (5, 5), (8, None)]. Raises ValueError on malformed input."""
    ranges: list[tuple[int, int | None]] = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            raise ValueError(f"empty item in --pages {spec!r}")
        lo_s, dash, hi_s = part.partition("-")
        try:
            lo = int(lo_s)
            hi = (int(hi_s) if hi_s else None) if dash else lo
        except ValueError:
            raise ValueError(f"cannot read {part!r} in --pages {spec!r}; use e.g. 1-5,8") from None
        if lo < 1 or (hi is not None and hi < lo):
            raise ValueError(f"invalid range {part!r} in --pages {spec!r}")
        ranges.append((lo, hi))
    return ranges


def select_pages(spec: str | None, total: int) -> list[int]:
    """1-based page numbers to process, in ascending order, clipped to ``total``."""
    if not spec:
        return list(range(1, total + 1))
    picked: set[int] = set()
    for lo, hi in parse_pages(spec):
        picked.update(range(lo, min(hi if hi is not None else total, total) + 1))
    return sorted(picked)
