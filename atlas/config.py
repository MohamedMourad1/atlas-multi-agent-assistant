"""
Configuration settings for Atlas Autonomous Multi-Agent Research Assistant.
"""

import os
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
KB_DIR = BASE_DIR / "atlas" / "knowledge_base" / "sample_corpus"
CHROMA_PERSIST_DIR = BASE_DIR / "data" / "chroma_db"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(KB_DIR, exist_ok=True)
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)


class Settings(BaseModel):
    """System-wide configuration settings."""

    # Project metadata
    project_name: str = "Atlas — Autonomous Multi-Agent Research & Report Assistant"
    version: str = "1.0.0"
    
    # LLM Provider Configuration
    # Supported: "gemini", "openai", "groq", "ollama", "mock"
    default_provider: str = Field(default_factory=lambda: os.getenv("ATLAS_LLM_PROVIDER", "gemini"))
    
    # API Keys
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", os.getenv("GOOGLE_API_KEY", "")))
    openai_api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    tavily_api_key: str = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    
    # Model Names
    gemini_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    openai_model: str = Field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    
    # Agent Hyperparameters
    max_repair_iterations: int = 3
    min_subquestions: int = 3
    max_subquestions: int = 6
    max_evidence_per_subquestion: int = 5
    critic_approval_threshold: float = 0.85
    
    # Tool Configs
    enable_web_search: bool = True
    enable_rag_kb: bool = True
    web_search_max_results: int = 4
    rag_top_k: int = 4
    
    # Paths
    base_dir: Path = BASE_DIR
    data_dir: Path = DATA_DIR
    kb_dir: Path = KB_DIR
    chroma_persist_dir: Path = CHROMA_PERSIST_DIR


settings = Settings()
