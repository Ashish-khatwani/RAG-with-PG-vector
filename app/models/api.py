from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class SourceMetadata(BaseModel):
    file_name: str
    file_type: str
    upload_timestamp: datetime | None = None
    chunk_index: int


class SourceChunk(BaseModel):
    chunk_id: UUID
    document_id: UUID
    content: str
    metadata: SourceMetadata
    retrieval_score: float = Field(default=0.0)
    rerank_score: float | None = None


class QueryFilters(BaseModel):
    file_name: str | None = None
    file_type: str | None = None


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    filters: QueryFilters | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    stream: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class IngestedDocumentResult(BaseModel):
    document_id: UUID
    file_name: str
    file_type: str
    chunk_count: int
    status: Literal["ingested", "skipped", "updated"]
    content_hash: str


class IngestResponse(BaseModel):
    documents: list[IngestedDocumentResult]


class DeleteDocumentRequest(BaseModel):
    document_id: UUID | None = None
    file_name: str | None = None


class DeleteDocumentResponse(BaseModel):
    deleted: bool
    message: str


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    database: str
    embedding_model: str
    reranker_model: str
    llm_provider: str
    details: dict[str, str]
