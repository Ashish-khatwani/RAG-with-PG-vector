from __future__ import annotations

from app.models.api import HealthResponse
from app.services.embeddings import EmbeddingService
from app.services.llm import LLMProvider
from app.services.repository import DocumentRepository
from app.services.reranker import RerankerService


class HealthService:
    def __init__(
        self,
        repository: DocumentRepository,
        embeddings: EmbeddingService,
        reranker: RerankerService,
        llm_provider: LLMProvider,
    ) -> None:
        self._repository = repository
        self._embeddings = embeddings
        self._reranker = reranker
        self._llm_provider = llm_provider

    async def check(self) -> HealthResponse:
        details: dict[str, str] = {}
        status = "healthy"

        try:
            document_count = await self._repository.document_count()
            chunk_count = await self._repository.chunk_count()
            details["database_counts"] = f"documents={document_count},chunks={chunk_count}"
            database_state = "healthy"
        except Exception as exc:
            database_state = "unhealthy"
            status = "unhealthy"
            details["database_error"] = str(exc)

        try:
            await self._embeddings.load()
            embedding_state = "healthy"
        except Exception as exc:
            embedding_state = "unhealthy"
            status = "unhealthy"
            details["embedding_error"] = str(exc)

        try:
            await self._reranker.load()
            reranker_state = "healthy"
        except Exception as exc:
            reranker_state = "unhealthy"
            status = "unhealthy"
            details["reranker_error"] = str(exc)

        try:
            llm_state = await self._llm_provider.healthcheck()
        except Exception as exc:
            llm_state = "degraded"
            if status == "healthy":
                status = "degraded"
            details["llm_error"] = str(exc)

        return HealthResponse(
            status=status,
            database=database_state,
            embedding_model=embedding_state,
            reranker_model=reranker_state,
            llm_provider=llm_state,
            details=details,
        )
