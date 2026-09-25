"""LLM-as-judge for OCR output: Claude looks at the page image and grades what Surya extracted.

PRIVACY: every sampled page image and its OCR text are sent to the Anthropic API. Use it on public or non-sensitive
documents only. Nothing else in docpipe makes network calls. The key is read from ANTHROPIC_API_KEY, never stored.
"""

from __future__ import annotations

import base64
import json
import os
import re
import statistics
import time
from pathlib import Path

from .metrics import cer, wer
from .schema import DocumentResult

DEFAULT_MODEL = "claude-sonnet-5"
MAX_IMAGE_SIDE = 1568  # larger images are downscaled by the API anyway

PROMPT = """You are grading an OCR system. The image is one scanned document page. Below is the text the OCR system extracted from it.

Do two things:
1. Transcribe ALL text visible on the page exactly as printed, in natural reading order, in its original script(s). Do not translate, do not correct spelling, and do not copy from the OCR text.
2. Judge the OCR text against the page.

Reply with ONLY one JSON object, no markdown fences, no other text:
{"transcription": "<full page text>",
 "score": <integer 1-10, 10 = perfect, 1 = unusable>,
 "verdict": "good" | "acceptable" | "poor",
 "reading_order_ok": <true|false>,
 "missing_text": ["<short excerpt of text on the page that the OCR text lacks>", ...],
 "wrong_text": [{"ocr": "<what the OCR text says>", "actual": "<what the page says>"}, ...],
 "hallucinated_text": ["<OCR text that is not on the page>", ...],
 "notes": "<one or two sentences>"}
List at most 10 items in wrong_text and 10 in missing_text.

OCR TEXT:
<<<
__OCR__
>>>"""


def sample_indices(n: int, k: int) -> list[int]:
    """Deterministic, evenly spread positions: the first, the last and the ones between."""
    if n <= k:
        return list(range(n))
    if k == 1:
        return [0]
    return sorted({round(i * (n - 1) / (k - 1)) for i in range(k)})


def _render(pdf_path: Path, page_number: int) -> bytes:
    import pymupdf

    with pymupdf.open(pdf_path) as doc:
        page = doc[page_number - 1]
        longest_in = max(page.rect.width, page.rect.height) / 72
        dpi = max(50, min(200, int(MAX_IMAGE_SIDE / longest_in)))
        return page.get_pixmap(dpi=dpi, alpha=False).tobytes("png")


def _parse(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in the judge's reply")
    return json.loads(text[start : end + 1])


def judge_page(client, model: str, png: bytes, ocr_text: str) -> dict:
    started = time.perf_counter()
    message = client.messages.create(
        model=model,
        max_tokens=16000,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png).decode()}},
                    {"type": "text", "text": PROMPT.replace("__OCR__", ocr_text)},
                ],
            }
        ],
    )
    latency = round(time.perf_counter() - started, 2)
    usage = {"latency_s": latency, "input_tokens": message.usage.input_tokens, "output_tokens": message.usage.output_tokens}
    if message.stop_reason == "max_tokens":
        raise ValueError(f"the judge's reply was cut off (page too dense for max_tokens); {latency}s, {usage['input_tokens']} in / {usage['output_tokens']} out tokens")
    verdict = _parse("".join(b.text for b in message.content if b.type == "text"))
    transcription = verdict.get("transcription", "")
    verdict.update(usage)
    verdict["cer_vs_transcription"] = round(cer(transcription, ocr_text), 4)
    verdict["wer_vs_transcription"] = round(wer(transcription, ocr_text), 4)
    return verdict


def _find_pdf(document: DocumentResult, pdfs_dir: Path | None) -> Path | None:
    for candidate in (Path(document.source), (pdfs_dir / Path(document.source).name) if pdfs_dir else None):
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def run(output_dir: Path, pdfs_dir: Path | None = None, pages_per_pdf: int = 3, model: str = DEFAULT_MODEL, say=print, report_dir: Path | None = None) -> dict:
    """Judge ``pages_per_pdf`` OCR pages of every PDF result under ``output_dir``. Writes judge_report.json there, or
    ``<report_dir>/judge_report_<output_dir name>.json`` when ``report_dir`` is given."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set. Create a key at console.anthropic.com, run `setx ANTHROPIC_API_KEY ...` in your own terminal, then open a new terminal.")
    try:
        import anthropic
    except ImportError:
        raise SystemExit("the judge needs the anthropic package: pip install anthropic") from None
    client = anthropic.Anthropic()

    documents = sorted(output_dir.rglob("document.json"))
    plan = []
    for path in documents:
        document = DocumentResult.read(path)
        if document.format != "pdf" or document.status == "error":
            continue
        ocr_units = [u for u in document.units if u.origin == "ocr" and u.status == "ok" and u.page]
        chosen = [ocr_units[i] for i in sample_indices(len(ocr_units), pages_per_pdf)]
        plan.append((path, document, chosen, len(ocr_units)))
    calls = sum(len(c) for _, _, c, _ in plan)
    say(f"judge: {model}; {calls} page(s) from {len(plan)} PDF(s) will be sent to the Anthropic API")

    report = {"model": model, "pages_per_pdf": pages_per_pdf, "documents": []}
    for path, document, chosen, n_ocr in plan:
        entry = {"document": path.parent.name, "source": document.source, "ocr_pages": n_ocr, "judged": []}
        pdf = _find_pdf(document, pdfs_dir)
        if pdf is None:
            entry["error"] = "source PDF not found; pass --pdfs"
        elif not chosen:
            entry["error"] = "no OCR pages in this document (text layer was used); nothing to judge"
        else:
            for unit in chosen:
                item = {"page": unit.page}
                say(f"  {entry['document']} p{unit.page}: calling {model} ...")
                try:
                    item.update(judge_page(client, model, _render(pdf, unit.page), unit.text))
                except Exception as exc:  # one failed page must not stop the run
                    item["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                entry["judged"].append(item)
                say(f"  {entry['document']} p{unit.page}: " + (item.get("error") or f"score {item.get('score')} {item.get('verdict')} CER {item['cer_vs_transcription']} WER {item['wer_vs_transcription']} | {item['latency_s']}s, {item['input_tokens']} in / {item['output_tokens']} out tokens"))
        ok = [j for j in entry["judged"] if "error" not in j]
        if ok:
            entry["mean_score"] = round(statistics.mean(j["score"] for j in ok), 2)
            entry["mean_cer"] = round(statistics.mean(j["cer_vs_transcription"] for j in ok), 4)
            entry["mean_wer"] = round(statistics.mean(j["wer_vs_transcription"] for j in ok), 4)
            entry["total_input_tokens"] = sum(j["input_tokens"] for j in ok)
            entry["total_output_tokens"] = sum(j["output_tokens"] for j in ok)
            entry["mean_latency_s"] = round(statistics.mean(j["latency_s"] for j in ok), 2)
        report["documents"].append(entry)

    scored = [d for d in report["documents"] if "mean_score" in d]
    report["overall"] = {
        "pdfs_judged": len(scored),
        "pages_judged": sum(len([j for j in d["judged"] if "error" not in j]) for d in scored),
        "mean_score": round(statistics.mean(d["mean_score"] for d in scored), 2) if scored else None,
        "mean_cer": round(statistics.mean(d["mean_cer"] for d in scored), 4) if scored else None,
        "mean_wer": round(statistics.mean(d["mean_wer"] for d in scored), 4) if scored else None,
        "total_input_tokens": sum(d["total_input_tokens"] for d in scored),
        "total_output_tokens": sum(d["total_output_tokens"] for d in scored),
        "total_latency_s": round(sum(j["latency_s"] for d in scored for j in d["judged"] if "error" not in j), 2),
    }
    if report_dir is None:
        target = output_dir / "judge_report.json"
    else:
        report_dir.mkdir(parents=True, exist_ok=True)
        target = report_dir / f"judge_report_{output_dir.resolve().name}.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    say(f"overall: {report['overall']}  -> {target}")
    return report


