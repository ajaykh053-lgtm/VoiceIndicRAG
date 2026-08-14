"""Application configuration using Pydantic Settings."""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API Keys
    GROQ_API_KEY: str = ""
    SARVAM_API_KEY: str = ""

    # Embedding & Retrieval
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIM: int = 384
    TOP_K: int = 5
    SIMILARITY_THRESHOLD: float = 0.25

    # Groq Model
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    GROQ_MAX_TOKENS: int = 512
    GROQ_TEMPERATURE: float = 0.2

    # Storage Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    INDEX_DIR: Path = DATA_DIR / "index"
    FAISS_INDEX_PATH: Path = INDEX_DIR / "faiss.index"
    METADATA_PATH: Path = INDEX_DIR / "metadata.json"
    SAMPLES_DATA_PATH: Path = DATA_DIR / "samples" / "msmarco_xi_sample.json"

    # Server Configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()

# Ensure directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.INDEX_DIR.mkdir(parents=True, exist_ok=True)
(settings.DATA_DIR / "samples").mkdir(parents=True, exist_ok=True)
