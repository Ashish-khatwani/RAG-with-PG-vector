from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.dependencies import get_ingestion_service, get_repository
from app.models.api import (
    DeleteDocumentRequest,
    DeleteDocumentResponse,
    IngestedDocumentResult,
)
from app.services.ingestion import IngestionService
from app.services.repository import DocumentRepository

router = APIRouter(tags=["documents"])


@router.delete("/document", response_model=DeleteDocumentResponse)
async def delete_document(
    request: DeleteDocumentRequest,
    repository: DocumentRepository = Depends(get_repository),
) -> DeleteDocumentResponse:
    deleted = await repository.delete_document(
        document_id=request.document_id,
        file_name=request.file_name,
    )
    return DeleteDocumentResponse(
        deleted=deleted,
        message="Document deleted." if deleted else "No matching document found.",
    )


@router.put("/document", response_model=IngestedDocumentResult)
async def update_document(
    file_name: str = Form(...),
    file: UploadFile | None = File(default=None),
    raw_text: str | None = Form(default=None),
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestedDocumentResult:
    content = await file.read() if file else None
    return await service.ingest_replacement(file_name=file_name, content=content, raw_text=raw_text)
