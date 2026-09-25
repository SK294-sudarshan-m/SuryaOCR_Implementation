"""Is a PDF text layer trustworthy, or does it only look fine on screen?

Pure Python, no dependencies. Nothing here can prove a text layer is *right*; it can only catch the failure
patterns seen in practice: private-use glyph mappings, legacy-font mojibake, and Indic text whose combining
marks were detached from their base letters during extraction (measured on a MuPDF-generated Hindi PDF).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

# Brahmic scripts Surya reads: Devanagari..Malayalam (Bengali also covers Assamese), Ol Chiki,
# Meetei Mayek, Devanagari/Vedic extensions.
INDIC_RANGES = (
    (0x0900, 0x0D7F),
    (0x1C50, 0x1C7F),
    (0x1CD0, 0x1CFF),
    (0xA8E0, 0xA8FF),
    (0xAAE0, 0xAAFF),
    (0xABC0, 0xABFF),
)
# Urdu, Kashmiri, Sindhi. Presentation forms are what many Urdu PDFs extract as.
ARABIC_RANGES = ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))
# Punctuation, currency (incl. the rupee sign), maths, arrows, bullets: expected in any script.
COMMON_RANGES = (
    (0x00A1, 0x00BF),
    (0x00D7, 0x00D7),
    (0x00F7, 0x00F7),
    (0x2000, 0x206F),
    (0x20A0, 0x20CF),
    (0x2100, 0x214F),
    (0x2190, 0x22FF),
    (0x25A0, 0x25FF),
    (0x2700, 0x27BF),
)
PUA_RANGES = ((0xE000, 0xF8FF), (0xF0000, 0xFFFFD), (0x100000, 0x10FFFD))
JOINERS = ("‌", "‍")

# Fonts whose extracted text is known to be mojibake (glyphs mapped onto ASCII / Latin-1). Lower-case
# substrings; deliberately short and unambiguous. ASCII-mapped text cannot be caught any other way.
LEGACY_INDIC_FONT_HINTS = (
    "krutidev",
    "kruti dev",
    "devlys",
    "chanakya",
    "walkman",
    "shivaji",
    "preeti",
    "shree-dev",
    "shreedev",
    "shree dev",
    "sutonnymj",
    "bamini",
    "aps-c-dv",
)


@dataclass(frozen=True)
class Thresholds:
    min_chars: int = 20  # fewer non-space characters than this is "no usable text layer"
    sparse_chars: int = 100  # with big images on the page, fewer than this is "sparse"
    image_cover: float = 0.5  # share of the page area covered by images that makes a page image-dominated
    min_image_cover: float = 0.05  # below this, an empty page counts as blank rather than image-only
    drawing_paths: int = 50  # vector paths on a text-free page that suggest outlined (glyph-path) text
    max_bad: float = 0.02  # private-use, U+FFFD, control characters
    max_foreign: float = 0.15  # characters outside the supported scripts and common punctuation
    max_orphan_ratio: float = 0.005  # combining marks with no base letter, per Indic character
    min_orphans: int = 2
    max_mixed_ratio: float = 0.02  # Indic words containing a stray foreign letter
    min_mixed: int = 2
    word_len_range: tuple[float, float] = (1.3, 25.0)


@dataclass
class Verdict:
    usable: bool
    reason: str
    metrics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _in(cp: int, ranges) -> bool:
    return any(lo <= cp <= hi for lo, hi in ranges)


@lru_cache(maxsize=None)
def classify(ch: str) -> str:
    """One of: space, bad, indic, arabic, latin, common, foreign."""
    cp = ord(ch)
    if ch.isspace():
        return "space"
    if ch in JOINERS:
        return "indic"
    cat = unicodedata.category(ch)
    if cat in ("Cc", "Cs", "Cn", "Co") or cp in (0xFFFD, 0xFFFE, 0xFFFF) or _in(cp, PUA_RANGES):
        return "bad"
    if _in(cp, INDIC_RANGES):
        return "indic"
    if _in(cp, ARABIC_RANGES):
        return "arabic"
    if 0x21 <= cp <= 0x7E:
        return "latin"
    if _in(cp, COMMON_RANGES):
        return "common"
    return "foreign"


@lru_cache(maxsize=None)
def _is_indic_mark(ch: str) -> bool:
    return classify(ch) == "indic" and unicodedata.category(ch) in ("Mn", "Mc")


@lru_cache(maxsize=None)
def _is_indic_base_or_mark(ch: str) -> bool:
    return classify(ch) == "indic" and (ch in JOINERS or unicodedata.category(ch)[0] in ("L", "M"))


def analyze_text(text: str) -> dict:
    counts = {"space": 0, "bad": 0, "indic": 0, "arabic": 0, "latin": 0, "common": 0, "foreign": 0}
    pua = replacement = 0
    for ch in text:
        kind = classify(ch)
        counts[kind] += 1
        if kind == "bad":
            if ord(ch) == 0xFFFD:
                replacement += 1
            elif unicodedata.category(ch) == "Co" or _in(ord(ch), PUA_RANGES):
                pua += 1

    n = sum(v for k, v in counts.items() if k != "space")

    orphans = 0
    for i, ch in enumerate(text):
        if _is_indic_mark(ch) and (i == 0 or not _is_indic_base_or_mark(text[i - 1])):
            orphans += 1

    tokens = text.split()
    indic_tokens = mixed = 0
    for token in tokens:
        kinds = {classify(c) for c in token}
        if "indic" in kinds:
            indic_tokens += 1
            if "foreign" in kinds:
                mixed += 1
    mean_word = sum(len(t) for t in tokens) / len(tokens) if tokens else 0.0

    return {
        "chars": n,
        "indic_chars": counts["indic"],
        "arabic_chars": counts["arabic"],
        "latin_chars": counts["latin"],
        "bad_pct": round(100 * counts["bad"] / n, 1) if n else 0.0,
        "private_use_pct": round(100 * pua / n, 1) if n else 0.0,
        "replacement_pct": round(100 * replacement / n, 1) if n else 0.0,
        "foreign_pct": round(100 * counts["foreign"] / n, 1) if n else 0.0,
        "orphan_marks": orphans,
        "mixed_script_words": mixed,
        "indic_words": indic_tokens,
        "words": len(tokens),
        "mean_word_len": round(mean_word, 1),
    }


def legacy_fonts(font_names) -> list[str]:
    return sorted({n for n in font_names if any(h in n.lower() for h in LEGACY_INDIC_FONT_HINTS)})


def judge_text_layer(text: str, font_names=(), t: Thresholds | None = None) -> Verdict:
    """Verdict on a text layer that has at least ``t.min_chars`` non-space characters."""
    t = t or Thresholds()
    m = analyze_text(text)
    n = m["chars"]
    problems: list[str] = []

    if m["bad_pct"] / 100 > t.max_bad:
        if m["private_use_pct"] >= m["replacement_pct"]:
            problems.append(f"private-use characters {m['private_use_pct']:.0f}%")
        if m["replacement_pct"] / 100 > t.max_bad / 2:
            problems.append(f"replacement characters U+FFFD {m['replacement_pct']:.0f}%")
        if not problems:
            problems.append(f"control or unassigned characters {m['bad_pct']:.0f}%")
    if m["foreign_pct"] / 100 > t.max_foreign:
        problems.append(f"characters outside the supported scripts {m['foreign_pct']:.0f}%")

    indic = m["indic_chars"]
    if indic and m["orphan_marks"] >= t.min_orphans and m["orphan_marks"] / indic > t.max_orphan_ratio:
        problems.append(f"{m['orphan_marks']} combining marks detached from their base letter")
    if m["indic_words"] and m["mixed_script_words"] >= t.min_mixed and (
        m["mixed_script_words"] / m["indic_words"] > t.max_mixed_ratio
    ):
        problems.append(f"{m['mixed_script_words']} Indic words contain stray non-Indic letters")

    # Word length only means something for scripts that put spaces between words.
    if m["words"] >= 5 and (m["indic_chars"] + m["latin_chars"]) / n > 0.5:
        lo, hi = t.word_len_range
        if m["mean_word_len"] > hi:
            problems.append(f"words implausibly long (mean {m['mean_word_len']} characters)")
        elif m["mean_word_len"] < lo:
            problems.append(f"words implausibly short (mean {m['mean_word_len']} characters)")

    legacy = legacy_fonts(font_names)
    if legacy:
        problems.append("legacy Indic font " + ", ".join(repr(f) for f in legacy))

    warnings = []
    if m["arabic_chars"] / max(n, 1) > 0.3:
        warnings.append(
            "Arabic-script text layer accepted without a structural check; if it reads reversed or "
            "disconnected, re-run with --force-ocr"
        )
    if problems:
        return Verdict(False, "text layer failed validity check (" + "; ".join(problems) + ")", m, warnings)
    return Verdict(True, "text layer passed validity check", m, warnings)
