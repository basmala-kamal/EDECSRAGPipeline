from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Gemini API
    gemini_api_key: str = ""
    gemini_embedding_model: str = "models/embedding-001"
    gemini_generation_model: str = "gemini-pro"

    # Chunking parameters
    chunk_size: int = 600
    chunk_overlap: int = 100

    # Retrieval parameters
    top_k: int = 5

    # ChromaDB
    chromadb_path: str = "./chroma_db"
    collection_name: str = "documents"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
