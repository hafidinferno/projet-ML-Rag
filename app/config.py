"""
Configuration module using Pydantic Settings.
Loads from .env file and environment variables.
"""

from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration with validation."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # === Ollama / LLM ===
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral"
    ollama_timeout: int = 120
    
    # === Embeddings ===
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    
    # === Paths ===
    docs_dir: Path = Path("./data/docs")
    vectordb_dir: Path = Path("./vectordb")
    logs_dir: Path = Path("./logs")
    
    # === RAG Parameters ===
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k_semantic: int = 5
    top_k_bm25: int = 3
    
    # === Hybrid Retrieval ===
    hybrid_semantic_weight: float = 0.7
    
    # === API ===
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    
    # === Security ===
    enable_injection_filter: bool = True
    
    @property
    def ollama_generate_url(self) -> str:
        """URL for Ollama generate endpoint."""
        return f"{self.ollama_base_url}/api/generate"
    
    @property
    def ollama_chat_url(self) -> str:
        """URL for Ollama chat endpoint."""
        return f"{self.ollama_base_url}/api/chat"


# Singleton instance
settings = Settings()
