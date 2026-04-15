from __future__ import annotations

import asyncpg


async def initialize_schema(pool: asyncpg.Pool, embedding_dimension: int) -> None:
    async with pool.acquire() as connection:
        await connection.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        await connection.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                file_name TEXT NOT NULL UNIQUE,
                file_type TEXT NOT NULL,
                content_hash TEXT NOT NULL UNIQUE,
                raw_text TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                upload_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        await connection.execute(
            """
            ALTER TABLE documents
            ADD COLUMN IF NOT EXISTS content_hash TEXT,
            ADD COLUMN IF NOT EXISTS raw_text TEXT NOT NULL DEFAULT '',
            ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
            ADD COLUMN IF NOT EXISTS upload_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            ADD COLUMN IF NOT EXISTS updated_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW();
            """
        )
        await connection.execute(
            """
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'documents'
                      AND column_name = 'uploaded_at'
                ) THEN
                    UPDATE documents
                    SET upload_timestamp = COALESCE(upload_timestamp, uploaded_at::timestamptz)
                    WHERE uploaded_at IS NOT NULL;
                END IF;
            END $$;
            """
        )
        await connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding VECTOR({embedding_dimension}) NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                content_tsv tsvector GENERATED ALWAYS AS (
                    to_tsvector('english', coalesce(content, ''))
                ) STORED,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE(document_id, chunk_index)
            );
            """
        )
        await connection.execute(
            """
            ALTER TABLE chunks
            ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb;
            """
        )
        await connection.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                      AND table_name = 'chunks'
                      AND column_name = 'content_tsv'
                ) THEN
                    ALTER TABLE chunks
                    ADD COLUMN content_tsv tsvector GENERATED ALWAYS AS (
                        to_tsvector('english', coalesce(content, ''))
                    ) STORED;
                END IF;
            END $$;
            """
        )
        await connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_file_name_unique
            ON documents (file_name);
            """
        )
        await connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_content_hash_unique
            ON documents (content_hash)
            WHERE content_hash IS NOT NULL;
            """
        )
        await connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_file_type
            ON documents (file_type);
            """
        )
        await connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_documents_metadata
            ON documents USING GIN (metadata);
            """
        )
        await connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id
            ON chunks (document_id);
            """
        )
        await connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_document_chunk_unique
            ON chunks (document_id, chunk_index);
            """
        )
        await connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_metadata
            ON chunks USING GIN (metadata);
            """
        )
        await connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv
            ON chunks USING GIN (content_tsv);
            """
        )
        await connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding
            ON chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
            """
        )
