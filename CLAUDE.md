# docpipe-surya
Documents in, `document.json` out (no .md output files; .md is for docs only). Native parsing for text; Surya v2 (llama.cpp, CPU) only for pixels.
Everything lives in this folder: put files in `input\`, results go to `output\` (contents git-ignored).

## Commands
- `.venv\Scripts\docpipe-surya process <paths> --out <dir> [--dpi 150] [--force-ocr] [--pages 1-5]`
- `.venv\Scripts\docpipe-surya doctor` | `evaluate doc.json --reference ref.txt` | `judge <out> [--pdfs DIR]`
- Install: see README. `pip install -e . --no-deps` after `pip install -r requirements.txt`.

## Gotchas
- NO test suite by user decision. Only verification so far: one real run on `samples\scansmpl.pdf`.
- Surya reads its settings from env at import: `model.configure_environment` must run before importing `surya`.
- Surya spawns `llama-server` (127.0.0.1) from `tools\llama.cpp`; GGUF weights in `models\surya-gguf` (not gated).
  `SURYA_INFERENCE_PARALLEL=1` (one page at a time). `SuryaEngine.close()` stops the server it started.
- Page cost grows steeply with dpi on CPU (96 dpi ~21 s, 150 ~60 s, 200 ~160 s for one text page): default 150.
- The backend start (~16 s) happens inside the first page's OCR call, so the first page's seconds include it.
- Native runs must not import torch/surya (lazy imports in `model.py`).
- Console: `cli._utf8_stdio()` forces UTF-8; JSON is written with `ensure_ascii=False`.
- Never read or print `ANTHROPIC_API_KEY`; `judge` reads it from the environment only.
