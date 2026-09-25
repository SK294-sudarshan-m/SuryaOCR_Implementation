"""Surya returns block content as HTML. These two helpers turn it into plain text and Markdown (stdlib only)."""

from __future__ import annotations

import re
from html.parser import HTMLParser

_HEADING = re.compile(r"h([1-6])")
_PARAS = {"p", "div", "blockquote", "pre", "figure", "figcaption"}
_BLOCKS = {"li", "tr"}


class _Converter(HTMLParser):
    """``markdown=False`` gives plain text (table cells joined with ' | '); ``True`` keeps tables as HTML."""

    def __init__(self, markdown: bool) -> None:
        super().__init__(convert_charrefs=True)
        self.md = markdown
        self.out: list[str] = []
        self.lists: list[list] = []  # [ordered?, counter]
        self.in_table = 0
        self.cell_open = False

    def _nl(self) -> None:
        if self.out and not self.out[-1].endswith("\n"):
            self.out.append("\n")

    def _para(self) -> None:
        if self.out:
            self.out.append("\n\n")

    def handle_starttag(self, tag, attrs):
        if self.md and self.in_table:
            self.out.append(self.get_starttag_text())
            self.in_table += tag == "table"
            return
        if tag == "table":
            self._para()
            self.in_table = 1
            if self.md:
                self.out.append(self.get_starttag_text())
        elif tag in ("td", "th"):
            if self.cell_open:
                self.out.append(" | ")
            self.cell_open = True
        elif tag == "tr":
            self._nl()
            self.cell_open = False
        elif (m := _HEADING.fullmatch(tag)):
            self._para()
            if self.md:
                self.out.append("#" * int(m.group(1)) + " ")
        elif tag in ("ul", "ol"):
            self._para()
            self.lists.append([tag == "ol", 0])
        elif tag == "li":
            self._nl()
            if self.lists:
                self.lists[-1][1] += 1
                ordered, n = self.lists[-1]
                self.out.append("  " * (len(self.lists) - 1) + (f"{n}. " if ordered else "- ") if self.md else "")
        elif tag == "br":
            self.out.append("\n")
        elif tag in ("b", "strong") and self.md:
            self.out.append("**")
        elif tag in ("i", "em") and self.md:
            self.out.append("*")
        elif tag == "math":
            self.out.append("$" if self.md else "")
        elif tag in _PARAS:
            self._para()
        elif tag in _BLOCKS:
            self._nl()

    def handle_endtag(self, tag):
        if self.md and self.in_table:
            self.out.append(f"</{tag}>")
            if tag == "table":
                self.in_table -= 1
                if not self.in_table:
                    self.out.append("\n")
            return
        if tag == "table":
            self.in_table = 0
            self._para()
        elif tag in ("b", "strong") and self.md:
            self.out.append("**")
        elif tag in ("i", "em") and self.md:
            self.out.append("*")
        elif tag == "math":
            self.out.append("$" if self.md else "")
        elif tag in ("ul", "ol"):
            if self.lists:
                self.lists.pop()
            self._para()
        elif tag in _PARAS or _HEADING.fullmatch(tag):
            self._para()
        elif tag in _BLOCKS:
            self._nl()

    def handle_data(self, data):
        self.out.append(data if self.in_table and self.md else re.sub(r"[ \t\r\n]+", " ", data))


def _convert(html: str, markdown: bool) -> str:
    conv = _Converter(markdown)
    conv.feed(html or "")
    conv.close()
    lines = [ln.strip() if not markdown or not ln.startswith("  ") else ln.rstrip() for ln in "".join(conv.out).split("\n")]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return text


def html_to_text(html: str) -> str:
    return _convert(html, False)


def html_to_markdown(html: str) -> str:
    """Paragraphs are separated by a blank line; tables stay HTML so merged cells survive."""
    return _convert(html, True)
