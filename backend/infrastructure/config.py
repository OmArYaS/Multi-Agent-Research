"""Application configuration loaded from environment / ``.env``."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Zero-cost, model-agnostic configuration.

    Defaults point at a local Ollama instance. ``OLLAMA_MODEL`` can be
    swapped (qwen2.5, gemma2, deepseek-r1, llama3.1, ...) without code
    changes because the LLM adapter is model-agnostic.
    """

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b-instruct-q4_k_m"
    max_iterations: int = 2
    db_path: str = "research_runs.db"
    search_max_results: int = 5

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}
