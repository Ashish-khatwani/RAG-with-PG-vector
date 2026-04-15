from __future__ import annotations

import asyncio
from pathlib import Path

import fitz
from pymupdf import FileDataError

from app.core.exceptions import UnsupportedFileTypeError, ValidationError
from app.models.domain import DocumentContent


class DocumentLoader:
    supported_extensions = {".pdf", ".txt", ".md", ".text"}

    async def from_upload(self, file_name: str, content: bytes) -> DocumentContent:
        suffix = Path(file_name).suffix.lower()
        if suffix not in self.supported_extensions:
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{suffix or 'unknown'}' for file '{file_name}'."
            )
        if not content:
            raise ValidationError(f"Uploaded file '{file_name}' is empty.")
        if suffix == ".pdf":
            text = await asyncio.to_thread(self._extract_pdf_text, content)
        else:
            text = content.decode("utf-8", errors="ignore")
        if not text.strip():
            raise ValidationError(f"Document '{file_name}' did not contain extractable text.")
        return DocumentContent(file_name=file_name, file_type=suffix.lstrip("."), text=text)

    def from_raw_text(self, file_name: str, text: str, file_type: str = "text") -> DocumentContent:
        if not text.strip():
            raise ValidationError(f"Raw text payload '{file_name}' is empty.")
        return DocumentContent(file_name=file_name, file_type=file_type, text=text)

    @staticmethod
    def _extract_pdf_text(content: bytes) -> str:
        if not content.lstrip().startswith(b"%PDF"):
            raise ValidationError("Uploaded file is not a valid PDF document.")
        try:
            document = fitz.open(stream=content, filetype="pdf")
        except (FileDataError, RuntimeError, ValueError) as exc:
            raise ValidationError("Uploaded file is not a valid PDF document.") from exc
        try:
            return "\n".join(page.get_text("text") for page in document)
        finally:
            document.close()
