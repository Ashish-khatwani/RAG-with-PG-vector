from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_query_service
from app.models.api import QueryRequest, QueryResponse
from app.services.query import QueryService

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    service: QueryService = Depends(get_query_service),
):
    if request.stream:
        return StreamingResponse(
            service.stream_answer(request),
            media_type="text/event-stream",
        )
    return await service.answer(request)
