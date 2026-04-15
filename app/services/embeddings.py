from __future__ import annotations

import asyncio
from typing import Sequence

from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str, batch_size: int, local_files_only: bool = False) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._local_files_only = local_files_only
        self._model: SentenceTransformer | None = None
        self._lock = asyncio.Lock()

    async def load(self) -> None:
        if self._model is not None:
            return
        async with self._lock:
            if self._model is None:
                self._model = await asyncio.to_thread(
                    SentenceTransformer,
                    self._model_name,
                    local_files_only=self._local_files_only,
                )

    async def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        await self.load()
        assert self._model is not None
        if not texts:
            return []
        embeddings = await asyncio.to_thread(
            self._model.encode,
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    async def embed_query(self, text: str) -> list[float]:
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def dimension(self) -> int:
        await self.load()
        assert self._model is not None
        return self._model.get_sentence_embedding_dimension()
