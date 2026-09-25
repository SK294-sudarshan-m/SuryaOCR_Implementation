"""Result types and their JSON form (document.json)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

from .config import ATTRIBUTION

# Optional Unit fields that are left out of the JSON when unset. OCR units always carry the OCR block below.
_OPTIONAL = (
    "page",
    "name",
    "page_width_pt",
    "page_height_pt",
    "text_layer",
    "seconds",
    "error_type",
    "message",
)
_OCR_KEYS = ("render_dpi", "width_px", "height_px", "blocks")


@dataclass
class Unit:
    index: int  # 0-based position in the document's unit list
    kind: str  # page | sheet | section | image
    origin: str  # native | ocr
    reason: str
    text: str = ""
    markdown: str = ""
    warnings: list[str] = field(default_factory=list)
    status: str = "ok"  # ok | error
    error_type: str | None = None
    message: str | None = None
    page: int | None = None  # 1-based, PDF pages and PPTX slides
    name: str | None = None  # sheet name, image label
    render_dpi: float | None = None  # null for image files, which are not rendered
    width_px: int | None = None
    height_px: int | None = None
    page_width_pt: float | None = None
    page_height_pt: float | None = None
    blocks: list[dict] | None = None  # Surya block records; boxes are in width_px x height_px
    text_layer: dict | None = None  # PDF: metrics behind the native/ocr decision
    seconds: float | None = None

    def to_dict(self) -> dict:
        d = {
            "index": self.index,
            "kind": self.kind,
            "origin": self.origin,
            "status": self.status,
            "reason": self.reason,
            "text": self.text,
            "markdown": self.markdown,
            "warnings": self.warnings,
        }
        for key in _OPTIONAL:
            if getattr(self, key) is not None:
                d[key] = getattr(self, key)
        if self.origin == "ocr":
            for key in _OCR_KEYS:
                d[key] = getattr(self, key) if getattr(self, key) is not None or key == "render_dpi" else None
            d["blocks"] = self.blocks if self.blocks is not None else []
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Unit:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class DocumentResult:
    source: str
    format: str | None = None
    sha256: str | None = None
    status: str = "ok"  # ok | partial | error
    units: list[Unit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    timings: dict = field(default_factory=dict)
    tool_versions: dict = field(default_factory=dict)
    error_type: str | None = None
    message: str | None = None
    model_attribution: str = ATTRIBUTION

    def to_dict(self) -> dict:
        d: dict = {
            "source": self.source,
            "sha256": self.sha256,
            "format": self.format,
            "status": self.status,
        }
        if self.status == "error":
            d["error_type"] = self.error_type
            d["message"] = self.message
        d["unit_count"] = len(self.units)
        d["units"] = [u.to_dict() for u in self.units]
        d["warnings"] = self.warnings
        d["timings"] = self.timings
        d["tool_versions"] = self.tool_versions
        d["model_attribution"] = self.model_attribution
        return d

    @classmethod
    def from_dict(cls, d: dict) -> DocumentResult:
        known = {f.name for f in fields(cls)} - {"units"}
        doc = cls(**{k: v for k, v in d.items() if k in known})
        doc.units = [Unit.from_dict(u) for u in d.get("units", [])]
        return doc

    def to_json(self, ensure_ascii: bool = False) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=2)

    @classmethod
    def read(cls, path: str | Path) -> DocumentResult:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
