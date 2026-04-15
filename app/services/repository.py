from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any
from uuid import UUID

import asyncpg

from app.core.exceptions import DocumentNotFoundError
from app.models.domain import ChunkRecord, DocumentRecord, RetrievedChunk
from app.utils.vector import vector_to_pg


class DocumentRepository:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def get_document_by_hash(self, content_hash: str) -> DocumentRecord | None:
        row = await self._pool.fetchrow(
            """
            SELECT id, file_name, file_type, content_hash, raw_text, metadata,
                   upload_timestamp, updated_timestamp
            FROM documents
            WHERE content_hash = $1
            """,
            content_hash,
        )
        return self._row_to_document(row) if row else None

    async def get_document_by_file_name(self, file_name: str) -> DocumentRecord | None:
        row = await self._pool.fetchrow(
            """
            SELECT id, file_name, file_type, content_hash, raw_text, metadata,
                   upload_timestamp, updated_timestamp
            FROM documents
            WHERE file_name = $1
            """,
            file_name,
        )
        return self._row_to_document(row) if row else None

    async def upsert_document(
        self,
        *,
        file_name: str,
        file_type: str,
        content_hash: str,
        raw_text: str,
        metadata: dict[str, Any],
    ) -> tuple[DocumentRecord, bool]:
        existing = await self.get_document_by_file_name(file_name)
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                row = await connection.fetchrow(
                    """
                    INSERT INTO documents (file_name, file_type, content_hash, raw_text, metadata)
                    VALUES ($1, $2, $3, $4, $5::jsonb)
                    ON CONFLICT (file_name)
                    DO UPDATE SET
                        file_type = EXCLUDED.file_type,
                        content_hash = EXCLUDED.content_hash,
                        raw_text = EXCLUDED.raw_text,
                        metadata = EXCLUDED.metadata,
                        updated_timestamp = NOW()
                    RETURNING id, file_name, file_type, content_hash, raw_text, metadata,
                              upload_timestamp, updated_timestamp,
                              (xmax = 0) AS inserted
                    """,
                    file_name,
                    file_type,
                    content_hash,
                    raw_text,
                    json.dumps(metadata),
                )
                await connection.execute("DELETE FROM chunks WHERE document_id = $1", row["id"])
        return self._row_to_document(row), existing is None

    async def insert_chunks(self, document_id: UUID, chunks: Sequence[ChunkRecord]) -> None:
        if not chunks:
            return
        async with self._pool.acquire() as connection:
            async with connection.transaction():
                await connection.executemany(
                    """
                    INSERT INTO chunks (document_id, chunk_index, content, embedding, metadata)
                    VALUES ($1, $2, $3, $4::vector, $5::jsonb)
                    """,
                    [
                        (
                            document_id,
                            chunk.chunk_index,
                            chunk.text,
                            vector_to_pg(chunk.embedding),
                            json.dumps(chunk.metadata),
                        )
                        for chunk in chunks
                    ],
                )

    async def delete_document(
        self, *, document_id: UUID | None = None, file_name: str | None = None
    ) -> bool:
        if document_id is None and file_name is None:
            raise DocumentNotFoundError("document_id or file_name is required")
        query = "DELETE FROM documents WHERE id = $1" if document_id else "DELETE FROM documents WHERE file_name = $1"
        value = document_id or file_name
        result = await self._pool.execute(query, value)
        return result.endswith("1")

    async def vector_search(
        self,
        embedding: list[float],
        limit: int,
        filters: dict[str, str | None],
    ) -> list[RetrievedChunk]:
        filter_sql, args = self._build_filters(filters, base_index=2)
        query = f"""
            SELECT c.id AS chunk_id, c.document_id, d.file_name, d.file_type, c.chunk_index,
                   c.content, c.metadata, d.upload_timestamp,
                   1 - (c.embedding <=> $1::vector) AS vector_score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE 1 = 1 {filter_sql}
            ORDER BY c.embedding <=> $1::vector
            LIMIT ${len(args) + 2}
        """
        rows = await self._pool.fetch(query, vector_to_pg(embedding), *args, limit)
        return [self._row_to_retrieved_chunk(row, vector_key="vector_score") for row in rows]

    async def keyword_search(
        self,
        query_text: str,
        limit: int,
        filters: dict[str, str | None],
    ) -> list[RetrievedChunk]:
        filter_sql, args = self._build_filters(filters, base_index=2)
        query = f"""
            SELECT c.id AS chunk_id, c.document_id, d.file_name, d.file_type, c.chunk_index,
                   c.content, c.metadata, d.upload_timestamp,
                   ts_rank_cd(c.content_tsv, websearch_to_tsquery('english', $1)) AS keyword_score
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.content_tsv @@ websearch_to_tsquery('english', $1) {filter_sql}
            ORDER BY keyword_score DESC
            LIMIT ${len(args) + 2}
        """
        rows = await self._pool.fetch(query, query_text, *args, limit)
        return [self._row_to_retrieved_chunk(row, keyword_key="keyword_score") for row in rows]

    async def document_count(self) -> int:
        return await self._pool.fetchval("SELECT COUNT(*) FROM documents")

    async def chunk_count(self) -> int:
        return await self._pool.fetchval("SELECT COUNT(*) FROM chunks")

    def _build_filters(
        self,
        filters: dict[str, str | None],
        *,
        base_index: int,
    ) -> tuple[str, list[str]]:
        clauses: list[str] = []
        args: list[str] = []
        if filters.get("file_name"):
            args.append(filters["file_name"] or "")
            clauses.append(f"AND d.file_name = ${base_index + len(args) - 1}")
        if filters.get("file_type"):
            args.append(filters["file_type"] or "")
            clauses.append(f"AND d.file_type = ${base_index + len(args) - 1}")
        return " " + " ".join(clauses) if clauses else "", args

    @staticmethod
    def _row_to_document(row: asyncpg.Record) -> DocumentRecord:
        return DocumentRecord(
            id=row["id"],
            file_name=row["file_name"],
            file_type=row["file_type"],
            content_hash=row["content_hash"],
            raw_text=row["raw_text"],
            metadata=dict(row["metadata"]),
            upload_timestamp=row["upload_timestamp"],
            updated_timestamp=row["updated_timestamp"],
        )

    @staticmethod
    def _row_to_retrieved_chunk(
        row: asyncpg.Record,
        *,
        vector_key: str | None = None,
        keyword_key: str | None = None,
    ) -> RetrievedChunk:
        metadata = dict(row["metadata"])
        metadata.setdefault("file_name", row["file_name"])
        metadata.setdefault("file_type", row["file_type"])
        metadata.setdefault("upload_timestamp", row["upload_timestamp"])
        metadata.setdefault("chunk_index", row["chunk_index"])
        return RetrievedChunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            file_name=row["file_name"],
            file_type=row["file_type"],
            chunk_index=row["chunk_index"],
            content=row["content"],
            metadata=metadata,
            vector_score=float(row[vector_key]) if vector_key and row[vector_key] is not None else None,
            keyword_score=float(row[keyword_key]) if keyword_key and row[keyword_key] is not None else None,
        )
