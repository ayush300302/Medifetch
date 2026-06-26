"""
Configuration management using pydantic-settings.
Loads configuration from environment variables or .env file.
"""

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings for Healthcare RAG Chatbot.
    """
    LLM_PROVIDER: str = Field(default="openai", description="Options: 'openai', 'gemini', or 'ollama'")
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini", description="OpenAI model name")
    OPENAI_BASE_URL: str = Field(default="", description="Optional custom base URL for OpenAI-compatible APIs")
    GEMINI_API_KEY: str = Field(default="", description="Google Gemini API key")
    OLLAMA_HOST: str = Field(default="http://localhost:11434", description="Local Ollama server host")
    OLLAMA_MODEL: str = Field(default="llama3", description="Model name in Ollama")
    
    # Embedding Settings
    EMBEDDING_MODEL: str = Field(
        default="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb",
        description="BioBERT HuggingFace model path"
    )
    
    # Reranker Settings
    RERANKER_MODEL: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-Encoder model for reranking"
    )
    
    # Vector DB
    VECTOR_STORE_PATH: str = Field(
        default="./vector_store/faiss_index",
        description="Directory where FAISS index and metadata are saved"
    )
    
    # Retrieval Hyperparameters
    TOP_K_RETRIEVAL: int = Field(default=10, description="Retrieve top K vectors from FAISS")
    TOP_K_RERANK: int = Field(default=3, description="Keep top N documents after reranking")
    RERANK_SCORE_THRESHOLD: float = Field(
        default=-3.0,
        description="Minimum Cross-Encoder score to retain chunk context"
    )

    
    # Environment
    APP_ENV: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")

    # Use env file if present
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Instantiate settings singleton
settings = Settings()

# Ensure vector store directory exists
os.makedirs(os.path.dirname(settings.VECTOR_STORE_PATH), exist_ok=True)
