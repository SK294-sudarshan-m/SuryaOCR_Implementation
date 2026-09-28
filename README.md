# SuryaOCR Implementation

Local command-line tool and Python library that turns documents into structured JSON. Anything that already
contains machine-readable text is **parsed natively**; only pixels — scans, photos, image-only PDF pages, pictures
embedded in Office files — go through **[Surya v2](https://github.com/datalab-to/surya) OCR**, running locally on
CPU via `llama.cpp`. No Markdown files are written; `.md` in this repo is documentation only.

> OCR by [Surya](https://github.com/datalab-to/surya) (Datalab), modified OpenRAIL-M weights.

## Contents

- [Licence](#licence-read-this)
- [Privacy and network use](#privacy-and-network-use)
- [Requirements](#requirements)
- [Install](#install-windows-python-310-cpu)
- [One-time backend and weights setup](#one-time-backend-and-weights-setup)
- [Usage](#usage)
- [Supported formats](#supported-formats)
- [Output](#output)
- [Known limits](#known-limits)

## Licence (read this)

Surya's code is Apache 2.0. The model weights are under a modified AI Pubs OpenRAIL-M licence: free for research,
personal use, and startups under $5M funding/revenue; a Datalab commercial licence is required otherwise. **Do not
offer this tool as a network service to third parties.**

## Privacy and network use

Processing makes no outbound network calls (Hugging Face is forced offline). On CPU, Surya starts `llama-server`
bound to `127.0.0.1` on a free port and talks to it over loopback only — this project exposes no server of its own.

**`judge` is the one exception:** it sends page images and OCR text to the Anthropic API. Use it only on
non-sensitive documents.

## Requirements

- Windows, Python 3.10+, CPU only (no GPU support has been tested)
- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) release binaries (Windows x64 CPU build)
- ~1.4 GB of disk space for the Surya v2 GGUF weights
- Optional: [LibreOffice](https://www.libreoffice.org/) (`soffice` on `PATH`) for legacy `.doc`/`.xls`/`.ppt` input
- Optional: an `ANTHROPIC_API_KEY` for the `judge` command

## Install (Windows, Python 3.10+, CPU)

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -e . --no-deps
```

Native formats need only PyMuPDF, python-docx, openpyxl, lxml and Pillow. Without `surya-ocr` installed, everything
still works except OCR.

## One-time backend and weights setup

1. **`llama-server`** — Surya does not fetch this itself. Download the official Windows x64 CPU build from the
   [llama.cpp releases page](https://github.com/ggml-org/llama.cpp/releases) (built and tested with `b11179`) and
   unzip it into `tools\llama.cpp`, or put it on `PATH`, or set the `LLAMA_CPP_BINARY` environment variable.
2. **Model weights** — not gated, no Hugging Face login required, about 1.4 GB:

   ```powershell
   .\.venv\Scripts\hf download datalab-to/surya-ocr-2-gguf surya-2.gguf surya-2-mmproj.gguf --local-dir models\surya-gguf
   ```

   The weights folder is resolved in this order: `--model-dir`, then `%DOCPIPE_MODEL_DIR%`, then
   `.\models\surya-gguf`.

3. Verify everything with:

   ```powershell
   .\.venv\Scripts\docpipe-surya doctor
   ```

## Usage

Put documents in `input\`; results are written to `output\` (contents of both are git-ignored, the folders
themselves are kept).

```powershell
docpipe-surya process input --out output
docpipe-surya process scan.pdf report.docx --out output --dpi 150 --pages 1-5
docpipe-surya doctor
docpipe-surya evaluate output\scan.pdf\document.json --reference truth.txt

# needs ANTHROPIC_API_KEY in the environment; sends page images to Anthropic
docpipe-surya judge output\ocr_output --pdfs input --pages-per-pdf 3 --report-dir output\judge_output
```

### `process` options

| Flag | Default | Meaning |
|---|---|---|
| `--out <dir>` | required | output folder |
| `--dpi <n>` | `150` | render resolution for pages sent to OCR |
| `--force-ocr` | off | OCR every page, even when a valid text layer exists |
| `--pages 1-5,8` | all | limit to specific pages |
| `--max-megapixels <n>` | `100` | reject/downscale images above this size |
| `--deterministic` | off | no timings in output, greedy decoding, fixed `llama-server` seed |
| `--device auto\|cpu\|cuda` | `auto` | inference device |
| `--model-dir <dir>` | see above | Surya weights folder |

Folders are searched recursively in sorted order. Progress is printed to `stderr` with `[mm:ss]` timestamps and a
heartbeat every 15 seconds while a page is processing. The process exits with code `1` if any document failed (the
batch always runs to completion).

## Supported formats

| Format | Handling |
|---|---|
| PDF | Per page: use the text layer if it passes validity checks, else OCR |
| PNG / JPEG / TIFF / BMP / WEBP | OCR |
| DOCX | Native text and tables; embedded images OCR'd |
| XLSX | Native cell values; embedded images OCR'd |
| PPTX | Native slide text and tables; embedded images OCR'd |
| Markdown | Native text; local image links OCR'd |
| DOC / XLS / PPT | Only if `soffice` is on `PATH` (converted first) |

Not supported: CSV, HTML, JSON, TXT, XML, ZIP, email, video.

## Output

Each input produces `<out>\<file name with extension>\document.json`. No Markdown files are written; each unit
carries its own text as Markdown in its `markdown` field.

Every unit has: `index`, `kind`, `origin`, `reason`, `status`, `text`, `markdown`, `warnings`, plus `page` and
`name` where applicable. PDF units also carry `text_layer` metrics used for the native-vs-OCR decision. OCR units
additionally carry `render_dpi`, `width_px`, `height_px`, and `blocks[]`, where each block has:

| Field | Meaning |
|---|---|
| `order` | reading order |
| `label`, `type` | block classification |
| `bbox_xyxy` | bounding box |
| `conf` | confidence (one page-level mean value from Surya, so identical across a page's blocks) |
| `text` | recognized text (`""` for failed or unread blocks, which are still reported with a warning) |
| `html`, `polygon` | extra detail from Surya |

## Known limits

- **CPU speed:** one dense English page (1273×1632 px, 150 DPI, 17 blocks) took 231 s, including a 16 s backend
  start-up (measured once). Shorter probe pages: ~21 s at 96 DPI, ~60 s at 150 DPI, ~163 s at 200 DPI. Batches of
  scans are slow — try `--dpi 96`–`120` if quality allows. Published reference: 0.108 pages/s on Apple Silicon.
- **Memory:** `llama-server` holds the model, ~2 GB resident in probes, plus ~0.35 GB for the Python process.
- **Indic-script quality is unverified** — no CER has been measured against ground truth.
- **Truncated blocks** cannot be detected directly; Surya only exposes an `error` flag, which is reported as-is.
- **Pillow** is pinned below version 11 by `surya-ocr`.
- **GPU (vLLM/Docker) path is untested.**
