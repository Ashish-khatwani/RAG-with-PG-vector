from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID


@dataclass(slots=True)
class DocumentContent:
    file_name: str
    file_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChunkRecord:
    chunk_index: int
    text: str
    embedding: list[float]
    metadata: dict[str, Any]


@dataclass(slots=True)
class DocumentRecord:
    id: UUID
    file_name: str
    file_type: str
    content_hash: str
    raw_text: str
    metadata: dict[str, Any]
    upload_timestamp: datetime
    updated_timestamp: datetime


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: UUID
    document_id: UUID
    file_name: str
    file_type: str
    chunk_index: int
    content: str
    metadata: dict[str, Any]
    vector_score: float | None = None
    keyword_score: float | None = None
    fusion_score: float = 0.0
    rerank_score: float | None = None
