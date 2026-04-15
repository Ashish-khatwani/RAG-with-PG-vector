from __future__ import annotations

import asyncpg

from app.core.config import Settings


class DatabasePool:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> asyncpg.Pool:
        if self._pool is None:
            self._pool = await asyncpg.create_pool(
                dsn=self._settings.postgres_dsn,
                min_size=self._settings.db_min_pool_size,
                max_size=self._settings.db_max_pool_size,
                command_timeout=self._settings.db_command_timeout,
            )
        return self._pool

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("Database pool has not been initialized.")
        return self._pool
