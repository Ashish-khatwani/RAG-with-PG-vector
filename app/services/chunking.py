from __future__ import annotations

from app.core.exceptions import ValidationError


class TextChunker:
    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_overlap >= chunk_size:
            raise ValidationError("chunk_overlap must be smaller than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        chunks: list[str] = []
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + self._chunk_size, text_length)
            if end < text_length:
                boundary = text.rfind(" ", start, end)
                if boundary > start + int(self._chunk_size * 0.6):
                    end = boundary
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= text_length:
                break
            start = max(end - self._chunk_overlap, 0)
        return chunks
