import os
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Qdrant Configuration
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "research_data")
    QDRANT_PREFER_GRPC: bool = os.getenv("QDRANT_PREFER_GRPC", "true").lower() == "true"
    
    # Embeddings Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "thenlper/gte-large")
    
    # LLM Configuration
    SAMBANOVA_API_KEY: Optional[str] = os.getenv("SAMBANOVA_API_KEY")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "DeepSeek-R1")
    
    # API Configuration
    API_PREFIX: str = "/api"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

settings = Settings()
