"""Failures that become per-document (or per-unit) error records instead of stopping a batch."""

from __future__ import annotations


class DocpipeError(Exception):
    """``error_type`` is what lands in document.json; it defaults to the class name."""

    def __init__(self, message: str, error_type: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type or type(self).__name__


class UnsupportedFormat(DocpipeError):
    pass


class CorruptFile(DocpipeError):
    pass


class EmptyFile(DocpipeError):
    pass


class PasswordProtected(DocpipeError):
    pass


class ImageTooLarge(DocpipeError):
    pass


class ModelUnavailable(DocpipeError):
    pass
