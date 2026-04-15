from __future__ import annotations

import json
from collections.abc import AsyncIterator

from app.models.api import QueryRequest, QueryResponse, SourceChunk, SourceMetadata
from app.models.domain import RetrievedChunk
from app.prompts.qa import build_qa_prompt
from app.services.llm import LLMProvider
from app.services.reranker import RerankerService
from app.services.retrieval import RetrievalService


class QueryService:
    def __init__(
        self,
        retrieval: RetrievalService,
        reranker: RerankerService,
        llm_provider: LLMProvider,
        reranker_top_k: int,
        answer_max_context_chunks: int,
    ) -> None:
        self._retrieval = retrieval
        self._reranker = reranker
        self._llm_provider = llm_provider
        self._reranker_top_k = reranker_top_k
        self._answer_max_context_chunks = answer_max_context_chunks

    async def answer(self, request: QueryRequest) -> QueryResponse:
        sources = await self._retrieve_sources(request)
        prompt = build_qa_prompt(request.query, self._build_context_blocks(sources))
        answer = await self._llm_provider.generate(prompt)
        return QueryResponse(answer=answer, sources=self._map_sources(sources))

    async def stream_answer(self, request: QueryRequest) -> AsyncIterator[str]:
        sources = await self._retrieve_sources(request)
        prompt = build_qa_prompt(request.query, self._build_context_blocks(sources))
        yield self._sse_event(
            "sources",
            {"sources": [source.model_dump(mode="json") for source in self._map_sources(sources)]},
        )
        async for token in self._llm_provider.stream_generate(prompt):
            yield self._sse_event("token", {"token": token})
        yield self._sse_event("done", {"status": "completed"})

    async def _retrieve_sources(self, request: QueryRequest) -> list[RetrievedChunk]:
        filters = request.filters.model_dump() if request.filters else {}
        retrieved = await self._retrieval.hybrid_search(
            request.query,
            top_k=request.top_k or (self._reranker_top_k * 2),
            filters=filters,
        )
        reranked = await self._reranker.rerank(request.query, retrieved[: self._reranker_top_k * 2])
        return reranked[: self._answer_max_context_chunks]

    def _build_context_blocks(self, sources: list[RetrievedChunk]) -> list[str]:
        blocks: list[str] = []
        for source in sources:
            blocks.append(f"[Source: {source.file_name} | chunk {source.chunk_index}]\n{source.content}")
        return blocks

    def _map_sources(self, sources: list[RetrievedChunk]) -> list[SourceChunk]:
        return [
            SourceChunk(
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                content=source.content,
                metadata=SourceMetadata(
                    file_name=source.file_name,
                    file_type=source.file_type,
                    upload_timestamp=source.metadata.get("upload_timestamp"),
                    chunk_index=source.chunk_index,
                ),
                retrieval_score=source.fusion_score,
                rerank_score=source.rerank_score,
            )
            for source in sources
        ]

    @staticmethod
    def _sse_event(event: str, payload: dict[str, object]) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"
