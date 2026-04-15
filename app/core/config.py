from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Document Intelligence Engine"
    app_env: str = Field(default="development")
    app_host: str = Field(default="0.0.0.0")
    app_port: int = Field(default=8000)
    log_level: str = Field(default="INFO")

    postgres_dsn: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/rag_backend"
    )
    db_min_pool_size: int = Field(default=1, ge=1)
    db_max_pool_size: int = Field(default=10, ge=1)
    db_command_timeout: int = Field(default=60, ge=5)

    embedding_model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    reranker_model_name: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    embedding_batch_size: int = Field(default=32, ge=1)
    reranker_top_k: int = Field(default=8, ge=1)
    retrieval_limit: int = Field(default=24, ge=1)
    chunk_size: int = Field(default=900, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)
    models_local_files_only: bool = Field(default=True)

    llm_provider: Literal["ollama", "mistral"] = Field(default="mistral")
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.1:8b")
    mistral_base_url: str = Field(default="https://api.mistral.ai")
    mistral_api_key: str = Field(default="")
    mistral_model: str = Field(default="mistral-small-latest")
    llm_timeout_seconds: int = Field(default=180, ge=10)
    answer_max_context_chunks: int = Field(default=6, ge=1)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
