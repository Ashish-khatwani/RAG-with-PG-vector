from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from app.core.exceptions import ValidationError
from app.models.api import IngestedDocumentResult
from app.models.domain import ChunkRecord, DocumentContent
from app.services.chunking import TextChunker
from app.services.document_loader import DocumentLoader
from app.services.embeddings import EmbeddingService
from app.services.repository import DocumentRepository
from app.services.text_cleaner import TextCleaner
from app.utils.hashing import sha256_text

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        loader: DocumentLoader,
        cleaner: TextCleaner,
        chunker: TextChunker,
        embeddings: EmbeddingService,
        repository: DocumentRepository,
    ) -> None:
        self._loader = loader
        self._cleaner = cleaner
        self._chunker = chunker
        self._embeddings = embeddings
        self._repository = repository

    async def ingest_uploads(
        self,
        uploads: list[tuple[str, bytes]],
        raw_texts: list[str] | None = None,
    ) -> list[IngestedDocumentResult]:
        if not uploads and not raw_texts:
            raise ValidationError("At least one file or raw_texts entry is required for ingestion.")
        documents: list[DocumentContent] = []
        for file_name, content in uploads:
            documents.append(await self._loader.from_upload(file_name, content))
        for index, raw_text in enumerate(raw_texts or [], start=1):
            generated_name = f"raw_text_{index}.txt"
            documents.append(self._loader.from_raw_text(generated_name, raw_text))
        return await asyncio.gather(*(self._ingest_document(content) for content in documents))

    async def ingest_replacement(
        self,
        *,
        file_name: str,
        content: bytes | None = None,
        raw_text: str | None = None,
    ) -> IngestedDocumentResult:
        if content is not None:
            document = await self._loader.from_upload(file_name, content)
        else:
            document = self._loader.from_raw_text(
                file_name=file_name,
                text=raw_text or "",
                file_type=Path(file_name).suffix.lstrip(".") or "text",
            )
        return await self._ingest_document(document, force_status="updated")

    async def _ingest_document(
        self,
        document: DocumentContent,
        force_status: str | None = None,
    ) -> IngestedDocumentResult:
        clean_text = self._cleaner.clean(document.text)
        content_hash = sha256_text(clean_text)

        existing_hash_match = await self._repository.get_document_by_hash(content_hash)
        if existing_hash_match and force_status is None:
            logger.info(
                "Skipping duplicate document",
                extra={
                    "incoming_file_name": document.file_name,
                    "existing_file_name": existing_hash_match.file_name,
                    "content_hash": content_hash,
                },
            )
            return IngestedDocumentResult(
                document_id=existing_hash_match.id,
                file_name=existing_hash_match.file_name,
                file_type=existing_hash_match.file_type,
                chunk_count=0,
                status="skipped",
                content_hash=existing_hash_match.content_hash,
            )

        chunks = self._chunker.chunk(clean_text)
        embeddings = await self._embeddings.embed_texts(chunks)
        timestamp = datetime.now(timezone.utc).isoformat()
        metadata = {
            "file_name": document.file_name,
            "file_type": document.file_type,
            "upload_timestamp": timestamp,
        }
        saved_document, inserted = await self._repository.upsert_document(
            file_name=document.file_name,
            file_type=document.file_type,
            content_hash=content_hash,
            raw_text=clean_text,
            metadata=metadata,
        )
        chunk_records = [
            ChunkRecord(
                chunk_index=index,
                text=chunk_text,
                embedding=embedding,
                metadata={**metadata, "chunk_index": index},
            )
            for index, (chunk_text, embedding) in enumerate(zip(chunks, embeddings, strict=True))
        ]
        await self._repository.insert_chunks(saved_document.id, chunk_records)
        status = force_status or ("ingested" if inserted else "updated")
        return IngestedDocumentResult(
            document_id=saved_document.id,
            file_name=saved_document.file_name,
            file_type=saved_document.file_type,
            chunk_count=len(chunk_records),
            status=status,
            content_hash=saved_document.content_hash,
        )
