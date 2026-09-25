# docpipe-surya

Local command-line tool and Python library that turns documents into structured text: for every input it writes one
`document.json` and one reading-ordered `document.md`. Anything that already contains machine-readable text is **parsed
natively**; only pixels (scans, photos, image-only PDF pages, pictures inside Office files) go through **Surya v2 OCR**.

**OCR by Surya (Datalab), modified OpenRAIL-M weights.**

## Licence (read this)
Surya's code is Apache 2.0. The model weights are under a modified AI Pubs OpenRAIL-M licence: free for research,
personal use and startups under $5M funding/revenue; otherwise a Datalab commercial licence is required. Do not offer this
tool as a network service to third parties.

## Privacy / network
Processing makes no outbound calls (Hugging Face is forced offline). On CPU, Surya starts `llama-server` bound to
`127.0.0.1` on a free port and talks to it over loopback only; this project exposes no server.
**`judge` is the exception:** it sends page images and OCR text to the Anthropic API. Use it on non-sensitive documents only.

## Install (Windows, Python 3.10+, CPU)
```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -e . --no-deps
```
Native formats need only PyMuPDF, python-docx, openpyxl, lxml and Pillow; without `surya-ocr` only OCR is unavailable.

### One-time backend and weights
1. `llama-server`: unzip the official Windows x64 CPU build from <https://github.com/ggml-org/llama.cpp/releases>
   into `tools\llama.cpp` (or put it on PATH, or set `LLAMA_CPP_BINARY`). Surya does not fetch it. Built with b11179.
2. Weights (not gated, no login, about 1.4 GB):
```powershell
.\.venv\Scripts\hf download datalab-to/surya-ocr-2-gguf surya-2.gguf surya-2-mmproj.gguf --local-dir models\surya-gguf
```
The folder is found via `--model-dir`, then `%DOCPIPE_MODEL_DIR%`, then `.\models\surya-gguf`. Then run `docpipe-surya doctor`.

## Use
```powershell
docpipe-surya process input\ --out out\
docpipe-surya process scan.pdf report.docx --out out\ --dpi 150 --pages 1-5
docpipe-surya doctor
docpipe-surya evaluate out\scan.pdf\document.json --reference truth.txt
docpipe-surya judge out\ --pdfs input\ --pages-per-pdf 3      # needs ANTHROPIC_API_KEY; sends images to Anthropic
```
`process` options: `--dpi` (default 150), `--force-ocr`, `--pages 1-5,8`, `--max-megapixels 100`, `--deterministic`
(no timings, greedy decoding, llama-server seed 0), `--device auto|cpu|cuda`, `--model-dir`. Folders are searched recursively
in sorted order. Progress goes to stderr with `[mm:ss]` prefixes and a heartbeat every 15 s while a page runs; the exit code
is 1 if any document failed (the batch always finishes).

## Formats
PDF (per page: text layer if it passes the validity check, else OCR), PNG/JPEG/TIFF/BMP/WEBP, DOCX, XLSX, PPTX, Markdown
(local images only), and DOC/XLS/PPT only when `soffice` is on PATH. No CSV, HTML, JSON, TXT, XML, ZIP, email or video.

## Output
`<out>\<file name with extension>\document.json` and `document.md` (units separated by `<!-- unit N | kind | ... | origin -->`).
Per unit: `index, kind, origin, reason, status, text, markdown, warnings`, plus `page`, `name`, PDF `text_layer` metrics, and for
OCR units `render_dpi, width_px, height_px, blocks[]` (`order, label, type, bbox_xyxy, conf, text`, plus `html`, `polygon`,
and `error`/`skipped` flags). Failed or unread blocks keep `text: ""` and produce a warning. Surya reports one page-level
mean confidence, so `conf` is the same for every block on a page.

## Known limits
- **CPU speed:** one dense English letter page (1273x1632 px, 150 dpi, 17 blocks) took 231 s including a 16 s backend start
  (one page only). Probes on a short generated page: about 21 s at 96 dpi, 60 s at 150, 163 s at 200. Batches of scans are
  slow; try `--dpi 96`-`120` if quality allows. Published Apple-Silicon figure: 0.108 pages/s.
- Memory: llama-server holds the model, about 2 GB resident in the probes, plus about 0.35 GB for Python.
- Indic-script quality is **unverified**: no CER was measured.
- Truncated blocks cannot be detected directly (Surya exposes only an `error` flag); failed blocks are reported.
- Pillow is pinned below 11 by surya-ocr. The GPU (vLLM/Docker) path is untested.
