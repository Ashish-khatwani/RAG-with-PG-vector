from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.dependencies import get_ingestion_service
from app.models.api import IngestResponse
from app.services.ingestion import IngestionService

router = APIRouter(tags=["ingestion"])


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    files: list[UploadFile] | None = File(default=None),
    raw_texts: list[str] | None = Form(default=None),
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    uploads: list[tuple[str, bytes]] = []
    for file in files or []:
        uploads.append((file.filename or "upload.txt", await file.read()))
    results = await service.ingest_uploads(uploads=uploads, raw_texts=raw_texts)
    return IngestResponse(documents=results)
