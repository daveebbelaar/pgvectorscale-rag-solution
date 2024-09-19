import os
import logging
from typing import Optional
from pydantic import Field, BaseModel
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv(dotenv_path="./.env")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class LLMProviderSettings(BaseModel):
    temperature: float = Field(default=0.0, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    max_retries: int = Field(default=3, ge=0)


class OpenAISettings(LLMProviderSettings):
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_model: str = Field(default="gpt-4")
    embedding_model: str = Field(default="text-embedding-3-small")


class DatabaseSettings(BaseModel):
    host: str = Field(default_factory=lambda: os.getenv("DB_HOST"))
    dbname: str = Field(default_factory=lambda: os.getenv("DB_NAME"))
    user: str = Field(default_factory=lambda: os.getenv("DB_USER"))
    password: str = Field(default_factory=lambda: os.getenv("DB_PASSWORD"))


class Settings(BaseModel):
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)


@lru_cache()
def get_settings() -> Settings:
    settings = Settings()
    setup_logging()
    return settings
