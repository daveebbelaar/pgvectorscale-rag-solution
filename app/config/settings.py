import logging
import os
from pathlib import Path
from datetime import timedelta
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")


def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


class LLMSettings(BaseModel):
    """Base settings for Language Model configurations."""

    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 3


class GroqSettings(LLMSettings):
    """Groq-specific LLM settings."""

    api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GROQ_API_KEY")
    )

    default_model: str = "llama-3.3-70b-versatile"


class EmbeddingSettings(BaseModel):
    """Embedding model settings."""

    model: str = "BAAI/bge-base-en-v1.5"
    dimensions: int = 768


class DatabaseSettings(BaseModel):
    """Database connection settings."""

    service_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("TIMESCALE_SERVICE_URL")
    )


class VectorStoreSettings(BaseModel):
    """Settings for the VectorStore."""

    table_name: str = "embeddings"
    embedding_dimensions: int = 768
    time_partition_interval: timedelta = timedelta(days=7)


class Settings(BaseModel):
    """Main application settings."""

    groq: GroqSettings = Field(default_factory=GroqSettings)

    embedding: EmbeddingSettings = Field(
        default_factory=EmbeddingSettings
    )

    database: DatabaseSettings = Field(
        default_factory=DatabaseSettings
    )

    vector_store: VectorStoreSettings = Field(
        default_factory=VectorStoreSettings
    )


@lru_cache()
def get_settings() -> Settings:
    """Create and return a cached Settings instance."""

    settings = Settings()
    setup_logging()

    return settings
