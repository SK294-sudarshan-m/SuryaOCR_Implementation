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
SCORE = "llm_quality_score_out_of_10"
VERDICT = "llm_verdict"
CERK = "character_error_rate_vs_llm_transcription"
WERK = "word_error_rate_vs_llm_transcription"
AVG_SCORE = "average_llm_quality_score_out_of_10"
AVG_CER = "average_character_error_rate_vs_llm_transcription"
AVG_WER = "average_word_error_rate_vs_llm_transcription"
RENAMED_REPLY_KEYS = {
    "transcription": "llm_transcription_of_page",
    "reading_order_ok": "llm_says_reading_order_is_correct",
    "missing_text": "text_on_page_missing_from_ocr",
    "wrong_text": "text_read_wrongly_by_ocr",
    "hallucinated_text": "text_in_ocr_not_on_page",
    "notes": "llm_notes",
}
SCORE_GUIDE = {
    SCORE: "Claude's own 1-10 rating of how faithful the OCR text is to the page image (10 = perfect, 1 = unusable). An LLM opinion, not ground truth.",
    VERDICT: "Claude's one-word summary of that rating: good, acceptable or poor.",
    CERK: "Character error rate of the OCR text measured against CLAUDE's transcription of the page (0 = identical, lower is better). Not a human reference.",
    WERK: "Word error rate against Claude's transcription (0 = identical, lower is better).",
    "judge_call_*": "Cost of the Claude API call that judged the page: latency in seconds and input/output token counts.",
    "average_*": "Averages over the judged pages of one PDF; the overall block averages the per-PDF averages, so every PDF counts equally.",
}
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
    usage = {"judge_call_latency_seconds": latency, "judge_call_input_tokens": message.usage.input_tokens, "judge_call_output_tokens": message.usage.output_tokens}
    if message.stop_reason == "max_tokens":
        raise ValueError(f"the judge's reply was cut off (page too dense for max_tokens); {latency}s, {usage['judge_call_input_tokens']} in / {usage['judge_call_output_tokens']} out tokens")
    verdict = _parse("".join(b.text for b in message.content if b.type == "text"))
    transcription = verdict.get("transcription", "")
    verdict.update(usage)
    for reply_key, stored_key in RENAMED_REPLY_KEYS.items():  # the model's short reply keys -> self-explanatory names
        if reply_key in verdict:
            verdict[stored_key] = verdict.pop(reply_key)
    verdict[SCORE] = verdict.pop("score", None)  # the model's reply key is "score"; stored under a self-explanatory name
    verdict[VERDICT] = verdict.pop("verdict", None)
    verdict[CERK] = round(cer(transcription, ocr_text), 4)
    verdict[WERK] = round(wer(transcription, ocr_text), 4)
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

    report = {"judge_model": model, "max_pages_judged_per_pdf": pages_per_pdf, "how_to_read_the_scores": SCORE_GUIDE, "per_pdf_results": []}
    for path, document, chosen, n_ocr in plan:
        entry = {"document_folder_name": path.parent.name, "source_pdf_as_recorded": document.source, "number_of_ocr_pages_in_pdf": n_ocr, "judged_pages": []}
        pdf = _find_pdf(document, pdfs_dir)
        if pdf is None:
            entry["error"] = "source PDF not found; pass --pdfs"
        elif not chosen:
            entry["error"] = "no OCR pages in this document (text layer was used); nothing to judge"
        else:
            for unit in chosen:
                item = {"page_number": unit.page}
                say(f"  {entry['document_folder_name']} p{unit.page}: calling {model} ...")
                try:
                    item.update(judge_page(client, model, _render(pdf, unit.page), unit.text))
                except Exception as exc:  # one failed page must not stop the run
                    item["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
                entry["judged_pages"].append(item)
                say(f"  {entry['document_folder_name']} p{unit.page}: " + (item.get("error") or f"LLM QUALITY SCORE {item.get(SCORE)}/10 ({item.get(VERDICT)}) | character error rate {item[CERK]} | word error rate {item[WERK]} | {item['judge_call_latency_seconds']}s, {item['judge_call_input_tokens']} in / {item['judge_call_output_tokens']} out tokens"))
        ok = [j for j in entry["judged_pages"] if "error" not in j and j.get(SCORE) is not None]
        if ok:
            entry[AVG_SCORE] = round(statistics.mean(j[SCORE] for j in ok), 2)
            entry[AVG_CER] = round(statistics.mean(j[CERK] for j in ok), 4)
            entry[AVG_WER] = round(statistics.mean(j[WERK] for j in ok), 4)
            entry["total_judge_call_input_tokens"] = sum(j["judge_call_input_tokens"] for j in ok)
            entry["total_judge_call_output_tokens"] = sum(j["judge_call_output_tokens"] for j in ok)
            entry["average_judge_call_latency_seconds"] = round(statistics.mean(j["judge_call_latency_seconds"] for j in ok), 2)
        if AVG_SCORE in entry:
            say(f"  ==> {entry['document_folder_name']}: AVERAGE LLM QUALITY SCORE {entry[AVG_SCORE]}/10 over {len(ok)} page(s) | average character error rate {entry[AVG_CER]} | average word error rate {entry[AVG_WER]}")
        report["per_pdf_results"].append(entry)

    scored = [d for d in report["per_pdf_results"] if AVG_SCORE in d]
    report["overall_summary"] = {
        "number_of_pdfs_judged": len(scored),
        "number_of_pages_judged": sum(len([j for j in d["judged_pages"] if "error" not in j]) for d in scored),
        AVG_SCORE: round(statistics.mean(d[AVG_SCORE] for d in scored), 2) if scored else None,
        AVG_CER: round(statistics.mean(d[AVG_CER] for d in scored), 4) if scored else None,
        AVG_WER: round(statistics.mean(d[AVG_WER] for d in scored), 4) if scored else None,
        "total_judge_call_input_tokens": sum(d["total_judge_call_input_tokens"] for d in scored),
        "total_judge_call_output_tokens": sum(d["total_judge_call_output_tokens"] for d in scored),
        "total_judge_call_latency_seconds": round(sum(j["judge_call_latency_seconds"] for d in scored for j in d["judged_pages"] if "error" not in j), 2),
    }
    if report_dir is None:
        target = output_dir / "judge_report.json"
    else:
        report_dir.mkdir(parents=True, exist_ok=True)
        target = report_dir / f"judge_report_{output_dir.resolve().name}.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    o = report["overall_summary"]
    say(f"==== OVERALL: average LLM quality score {o[AVG_SCORE]}/10 over {o['number_of_pages_judged']} page(s) of {o['number_of_pdfs_judged']} PDF(s) | average character error rate {o[AVG_CER]} | average word error rate {o[AVG_WER]} | {o['total_judge_call_input_tokens']} in / {o['total_judge_call_output_tokens']} out tokens")
    say(f"report: {target}")
    return report


