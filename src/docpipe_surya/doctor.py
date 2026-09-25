"""`docpipe-surya doctor`: what is installed, what will run, and one concrete fix per problem."""

from __future__ import annotations

import platform
import sys

from .config import GGUF_FILES, LICENCE_NOTE, Options, default_llama_server, resolve_device
from .model import Problem, llama_server_version, model_dir_problems, package_version, runtime_problems


def _size(path) -> str:
    return f"{path.stat().st_size / 2**20:,.0f} MB" if path.is_file() else "MISSING"


def run(opts: Options, out=None) -> int:
    out = out or sys.stdout
    say = lambda line="": print(line, file=out)  # noqa: E731
    model_dir = opts.resolved_model_dir()
    device = resolve_device(opts.device)
    binary = default_llama_server()

    say(f"python         {platform.python_version()}  ({sys.executable})")
    for name in ("surya-ocr", "torch", "torchvision", "transformers", "PyMuPDF", "python-docx", "openpyxl", "lxml", "Pillow"):
        say(f"{name:<14} {package_version(name) or 'NOT INSTALLED'}")
    say(f"device         {device}  (backend: {'llama.cpp' if device == 'cpu' else 'vLLM in Docker'})")
    say(f"llama-server   {binary or 'NOT FOUND'}" + (f"  [{llama_server_version(binary)}]" if binary else ""))
    say(f"model dir      {model_dir}")
    for name in GGUF_FILES:
        say(f"  {name:<20} {_size(model_dir / name)}")
    say("soffice        " + (__import__("shutil").which("soffice") or "not found (legacy .doc/.xls/.ppt will be reported unsupported)"))
    say(f"licence        {LICENCE_NOTE}")
    say("network        processing makes no outbound calls; Surya's llama-server listens on 127.0.0.1 only")

    problems: list[Problem] = runtime_problems(device, model_dir)
    say()
    if not problems:
        say("OK: OCR is ready.")
        return 0
    say(f"{len(problems)} problem(s):")
    for problem in problems:
        say(f"  - {problem.message}")
        say(f"    fix: {problem.fix}")
    return 1
