from __future__ import annotations

from app.models.domain import RetrievedChunk
from app.services.embeddings import EmbeddingService
from app.services.repository import DocumentRepository


class RetrievalService:
    def __init__(self, repository: DocumentRepository, embeddings: EmbeddingService) -> None:
        self._repository = repository
        self._embeddings = embeddings

    async def hybrid_search(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, str | None],
    ) -> list[RetrievedChunk]:
        query_embedding = await self._embeddings.embed_query(query)
        vector_hits = await self._repository.vector_search(query_embedding, top_k, filters)
        keyword_hits = await self._repository.keyword_search(query, top_k, filters)
        fused = self._reciprocal_rank_fusion(vector_hits, keyword_hits)
        return sorted(fused.values(), key=lambda item: item.fusion_score, reverse=True)[:top_k]

    @staticmethod
    def _reciprocal_rank_fusion(
        vector_hits: list[RetrievedChunk],
        keyword_hits: list[RetrievedChunk],
        constant: int = 60,
    ) -> dict[str, RetrievedChunk]:
        fused: dict[str, RetrievedChunk] = {}
        for hits, score_attr in ((vector_hits, "vector_score"), (keyword_hits, "keyword_score")):
            for rank, chunk in enumerate(hits, start=1):
                key = str(chunk.chunk_id)
                current = fused.get(key, chunk)
                current.fusion_score += 1.0 / (constant + rank)
                if score_attr == "vector_score":
                    current.vector_score = chunk.vector_score
                else:
                    current.keyword_score = chunk.keyword_score
                fused[key] = current
        return fused
