"""OCR evaluation metrics. Pure Python, no dependencies.

Lexical: CER, WER, token F1. Semantic: field accuracy, exact match. Layout: IoU, reading-order distance.
Operational: straight-through-processing / human-review rate, throughput.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Sequence

from .schema import DocumentResult


def normalize(text: str) -> str:
    """NFC, whitespace collapsed."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def edit_distance(a: Sequence, b: Sequence) -> int:
    """Levenshtein distance (insertions, deletions, substitutions) over any two sequences."""
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (x != y)))
        previous = current
    return previous[-1]


def cer(reference: str, hypothesis: str) -> float:
    """(S + D + I) / characters in the reference."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    return edit_distance(ref, hyp) / max(1, len(ref))


def wer(reference: str, hypothesis: str) -> float:
    """(S + D + I) / words in the reference; a word with any wrong character is a full error."""
    ref, hyp = normalize(reference).split(), normalize(hypothesis).split()
    return edit_distance(ref, hyp) / max(1, len(ref))


def token_f1(reference: str, hypothesis: str) -> dict:
    """Precision (noise), recall (missed words) and F1 over the multiset of words."""
    ref, hyp = Counter(normalize(reference).split()), Counter(normalize(hypothesis).split())
    overlap = sum((ref & hyp).values())
    precision = overlap / max(1, sum(hyp.values()))
    recall = overlap / max(1, sum(ref.values()))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def field_accuracy(expected: dict, actual: dict) -> float:
    """Correctly populated fields / total expected fields (values compared after normalisation)."""
    if not expected:
        return 1.0
    right = sum(1 for k, v in expected.items() if k in actual and normalize(str(actual[k])) == normalize(str(v)))
    return right / len(expected)


def exact_match_rate(pairs: Sequence[tuple[dict, dict]]) -> float:
    """Share of documents whose every expected field is correct. ``pairs``: (expected, actual) per document."""
    if not pairs:
        return 0.0
    return sum(1 for e, a in pairs if field_accuracy(e, a) == 1.0) / len(pairs)


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two [x0, y0, x1, y1] boxes."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def layout_iou(predicted: Sequence[Sequence[float]], truth: Sequence[Sequence[float]], threshold: float = 0.5) -> dict:
    """Greedy best-IoU matching. ``mean_iou`` averages over ground-truth boxes (unmatched count as 0);
    precision/recall count matches at IoU >= threshold."""
    unused = list(range(len(predicted)))
    scores, matched = [], 0
    for t in truth:
        best = max(unused, key=lambda i: iou(predicted[i], t), default=None)
        score = iou(predicted[best], t) if best is not None else 0.0
        scores.append(score)
        if score >= threshold:
            matched += 1
            unused.remove(best)
    return {
        "mean_iou": sum(scores) / max(1, len(truth)),
        "precision": matched / max(1, len(predicted)),
        "recall": matched / max(1, len(truth)),
    }


def reading_order_distance(predicted: Sequence, truth: Sequence) -> float:
    """Normalised edit distance between two orderings of the same block ids (0 = identical order).
    A flat sequence, not a tree: it catches column-by-column vs row-by-row mistakes but not nesting."""
    return edit_distance(list(predicted), list(truth)) / max(1, len(truth))


def needs_review(result: DocumentResult, min_conf: float = 0.8) -> list[str]:
    """Reasons a document cannot go straight through: failure, warnings, or low-confidence OCR blocks."""
    reasons = []
    if result.status != "ok":
        reasons.append(f"status {result.status}")
    for unit in result.units:
        if unit.warnings:
            reasons.append(f"unit {unit.index}: {unit.warnings[0]}")
        low = [b for b in (unit.blocks or []) if b.get("text") and b.get("conf", 1.0) < min_conf]
        if low:
            reasons.append(f"unit {unit.index}: {len(low)} block(s) below confidence {min_conf}")
    return reasons


def straight_through_rate(results: Sequence[DocumentResult], min_conf: float = 0.8) -> dict:
    """STP = share of documents with nothing needing review; HITL is the rest."""
    flagged = sum(1 for r in results if needs_review(r, min_conf))
    n = max(1, len(results))
    return {"stp_rate": 1 - flagged / n, "hitl_rate": flagged / n}


def throughput(result: DocumentResult) -> dict:
    """Pages (units) per second over the whole document and over OCR time only."""
    total = result.timings.get("total_seconds") or 0.0
    ocr = result.timings.get("ocr_seconds") or 0.0
    ocr_units = sum(1 for u in result.units if u.origin == "ocr")
    return {
        "units": len(result.units),
        "units_per_second": len(result.units) / total if total else None,
        "ocr_units_per_second": ocr_units / ocr if ocr else None,
        "peak_rss_mb": result.timings.get("peak_rss_mb"),
    }


def evaluate(result: DocumentResult, reference_text: str) -> dict:
    """Lexical, operational and (when OCR blocks exist) confidence figures for one document."""
    hypothesis = "\n\n".join(u.text for u in result.units if u.status == "ok" and u.text)
    return {
        "source": result.source,
        "cer": cer(reference_text, hypothesis),
        "wer": wer(reference_text, hypothesis),
        **{f"token_{k}": v for k, v in token_f1(reference_text, hypothesis).items()},
        **throughput(result),
        "needs_review": needs_review(result),
    }
