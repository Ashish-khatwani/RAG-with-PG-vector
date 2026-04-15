from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes.document import router as document_router
from app.api.routes.health import router as health_router
from app.api.routes.ingest import router as ingest_router
from app.api.routes.query import router as query_router
from app.core.config import get_settings
from app.core.exceptions import (
    AppError,
    DocumentNotFoundError,
    UnsupportedFileTypeError,
    ValidationError,
)
from app.core.logging import configure_logging
from app.db.pool import DatabasePool
from app.db.schema import initialize_schema
from app.services.chunking import TextChunker
from app.services.document_loader import DocumentLoader
from app.services.embeddings import EmbeddingService
from app.services.health import HealthService
from app.services.ingestion import IngestionService
from app.services.llm import LLMProvider, MistralProvider, OllamaProvider
from app.services.query import QueryService
from app.services.repository import DocumentRepository
from app.services.reranker import RerankerService
from app.services.retrieval import RetrievalService
from app.services.text_cleaner import TextCleaner

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)


def build_llm_provider() -> LLMProvider:
    if settings.llm_provider == "mistral":
        return MistralProvider(
            base_url=settings.mistral_base_url,
            api_key=settings.mistral_api_key,
            model=settings.mistral_model,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    return OllamaProvider(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    embedding_service = EmbeddingService(
        model_name=settings.embedding_model_name,
        batch_size=settings.embedding_batch_size,
        local_files_only=settings.models_local_files_only,
    )
    reranker_service = RerankerService(
        model_name=settings.reranker_model_name,
        local_files_only=settings.models_local_files_only,
    )
    await embedding_service.load()
    await reranker_service.load()

    database = DatabasePool(settings)
    pool = await database.connect()
    await initialize_schema(pool, await embedding_service.dimension())

    repository = DocumentRepository(pool)
    ingestion_service = IngestionService(
        loader=DocumentLoader(),
        cleaner=TextCleaner(),
        chunker=TextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        ),
        embeddings=embedding_service,
        repository=repository,
    )
    llm_provider = build_llm_provider()
    retrieval_service = RetrievalService(repository=repository, embeddings=embedding_service)
    query_service = QueryService(
        retrieval=retrieval_service,
        reranker=reranker_service,
        llm_provider=llm_provider,
        reranker_top_k=settings.reranker_top_k,
        answer_max_context_chunks=settings.answer_max_context_chunks,
    )
    health_service = HealthService(
        repository=repository,
        embeddings=embedding_service,
        reranker=reranker_service,
        llm_provider=llm_provider,
    )

    app.state.settings = settings
    app.state.database = database
    app.state.document_repository = repository
    app.state.ingestion_service = ingestion_service
    app.state.query_service = query_service
    app.state.health_service = health_service

    logger.info("Application startup complete", extra={"app_name": settings.app_name})
    try:
        yield
    finally:
        await database.close()
        logger.info("Application shutdown complete")


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Async backend service for ingestion, retrieval, reranking, and grounded QA.",
    lifespan=lifespan,
)

app.include_router(ingest_router)
app.include_router(query_router)
app.include_router(document_router)
app.include_router(health_router)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    status_code = 500
    if isinstance(exc, (ValidationError, UnsupportedFileTypeError)):
        status_code = 400
    if isinstance(exc, DocumentNotFoundError):
        status_code = 404
    logging.getLogger("app.errors").warning(
        "Application error",
        extra={"path": str(request.url.path), "error": str(exc)},
    )
    return JSONResponse(status_code=status_code, content={"detail": str(exc)})


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Document Intelligence Engine is running."}
