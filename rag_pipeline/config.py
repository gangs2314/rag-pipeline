"""Configuration management for RAG pipeline."""

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable override support."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Vector Store
    vector_store_type: Literal["chromadb", "qdrant"] = "chromadb"
    chromadb_path: Path = Path("./data/chromadb")
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    # Embedding Model
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_cache_dir: Path = Path("./models/embeddings")
    embedding_device: str = "cpu"  # "cuda" if GPU available

    # Chunking Strategy
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 50  # tokens
    use_semantic_chunking: bool = False
    use_parent_child_chunking: bool = True
    parent_chunk_size: int = 1024
    child_chunk_size: int = 256

    # Document Processing
    upload_dir: Path = Path("./data/uploads")
    max_file_size_mb: int = 100

    # Retrieval
    top_k: int = 5
    min_similarity_score: float = 0.0

    # Kaggle
    kaggle_datasets_dir: Path = Path("./data/kaggle_datasets")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()

# Create directories
settings.chromadb_path.mkdir(parents=True, exist_ok=True)
settings.embedding_cache_dir.mkdir(parents=True, exist_ok=True)
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.kaggle_datasets_dir.mkdir(parents=True, exist_ok=True)
