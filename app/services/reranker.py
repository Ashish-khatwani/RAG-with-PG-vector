from __future__ import annotations

import asyncio
from typing import Sequence

from sentence_transformers import CrossEncoder

from app.models.domain import RetrievedChunk


class RerankerService:
    def __init__(self, model_name: str, local_files_only: bool = False) -> None:
        self._model_name = model_name
        self._local_files_only = local_files_only
        self._model: CrossEncoder | None = None
        self._lock = asyncio.Lock()

    async def load(self) -> None:
        if self._model is not None:
            return
        async with self._lock:
            if self._model is None:
                self._model = await asyncio.to_thread(
                    CrossEncoder,
                    self._model_name,
                    local_files_only=self._local_files_only,
                )

    async def rerank(self, query: str, chunks: Sequence[RetrievedChunk]) -> list[RetrievedChunk]:
        await self.load()
        assert self._model is not None
        if not chunks:
            return []
        pairs = [[query, chunk.content] for chunk in chunks]
        scores = await asyncio.to_thread(self._model.predict, pairs)
        reranked: list[RetrievedChunk] = []
        for chunk, score in zip(chunks, scores, strict=True):
            chunk.rerank_score = float(score)
            reranked.append(chunk)
        return sorted(reranked, key=lambda item: item.rerank_score or 0.0, reverse=True)
