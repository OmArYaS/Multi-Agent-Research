"""Ollama LLM adapter implementing the domain ``LLMProvider`` interface."""

from __future__ import annotations

import json
import logging
from typing import List, Type

import httpx
from pydantic import BaseModel

from domain.interfaces import LLMProvider
from domain.models import BaseModel as _  # noqa: F401  (re-export convenience)
from infrastructure.parser import ParseError, StructuredOutputParser

logger = logging.getLogger(__name__)


class OllamaLLMProvider:
    """Concrete LLM provider backed by a local Ollama server."""

    def __init__(
        self,
        base_url: str,
        model: str,
        parser: StructuredOutputParser,
        timeout: float = 180.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.parser = parser
        self.timeout = timeout

    # -- low-level HTTP ---------------------------------------------------

    async def _generate_raw(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.3,
        json_mode: bool = False,
    ) -> str:
        payload: dict = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/api/generate", json=payload)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                logger.error(
                    "Ollama generate failed (%s): %s",
                    response.status_code,
                    response.text[:500],
                )
                raise
            data = response.json()
            return data.get("response", "")

    # -- LLMProvider interface -------------------------------------------

    async def generate(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.3,
        json_mode: bool = False,
    ) -> str:
        return await self._generate_raw(
            prompt=prompt,
            system=system,
            model=model,
            temperature=temperature,
            json_mode=json_mode,
        )

    async def generate_structured(
        self,
        prompt: str,
        system: str | None = None,
        model: str | None = None,
        schema: type[BaseModel] | None = None,
        temperature: float = 0.3,
        max_retries: int = 2,
    ) -> BaseModel:
        if schema is None:
            raise ValueError("schema is required for structured generation")

        async def regenerate(current_prompt: str) -> str:
            return await self._generate_raw(
                prompt=current_prompt,
                system=system,
                model=model,
                temperature=temperature,
                json_mode=True,
            )

        try:
            return await self.parser.parse_with_retry(
                regenerate=regenerate,
                model=schema,
                prompt=prompt,
                system=system,
                max_retries=max_retries,
            )
        except ParseError as exc:
            logger.error("Ollama structured generation failed: %s", exc)
            raise

    async def list_models(self) -> List[str]:
        """Return locally available Ollama model names (empty if offline)."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
            return [item.get("name", "") for item in data.get("models", []) if item.get("name")]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama offline or unreachable; returning empty model list: %s", exc)
            return []
