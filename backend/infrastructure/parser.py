"""Resilient structured-output parser for model-agnostic LLMs.

Strips model-specific noise (reasoning tags such as ``<think>`` and markdown
code fences) before Pydantic validation, with retry/fallback handling so that
different open-source Ollama models (Qwen, Gemma, DeepSeek, Llama, ...) work
reliably without Qwen-specific prompt templates or stop tokens.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

# Matches <think>...</think> blocks (case-insensitive, multiline).
THINK_TAG_RE = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.DOTALL | re.IGNORECASE)
# Matches fenced code blocks, capturing the inner content.
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
# Matches a balanced-ish JSON object/array substring.
JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


class ParseError(Exception):
    """Raised when output cannot be cleaned and validated after retries."""


class StructuredOutputParser:
    """Cleans raw model text and validates it against a Pydantic schema.

    Cleaning steps (in order):
      1. Remove ``<think>...</think>`` reasoning blocks.
      2. Extract fenced JSON (`` ```json ... ``) if present.
      3. Fall back to the first ``{ ... }`` substring.
      4. Validate with Pydantic.
    """

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries

    # -- cleaning ---------------------------------------------------------

    def clean(self, text: str) -> str:
        """Strip reasoning tags and fences, returning the JSON candidate."""
        if not text:
            return ""
        cleaned = THINK_TAG_RE.sub("", text)
        fence_match = FENCE_RE.search(cleaned)
        if fence_match:
            cleaned = fence_match.group(1)
        return cleaned.strip()

    def extract_json_object(self, text: str) -> str | None:
        """Best-effort extraction of the first JSON object substring."""
        match = JSON_OBJECT_RE.search(text)
        if not match:
            return None
        candidate = match.group(0)
        # Validate brace balance crudely to avoid grabbing too much.
        if candidate.count("{") != candidate.count("}"):
            return None
        return candidate

    # -- validation -------------------------------------------------------

    def parse(self, text: str, model: Type[T]) -> T:
        """Clean and validate ``text`` into ``model``.

        Raises :class:`ParseError` if validation fails.
        """
        cleaned = self.clean(text)
        last_error: Exception | None = None

        candidates = [cleaned]
        extracted = self.extract_json_object(cleaned)
        if extracted and extracted != cleaned:
            candidates.append(extracted)

        for candidate in candidates:
            try:
                return model.model_validate_json(candidate)
            except ValidationError as exc:
                last_error = exc
                try:
                    return model.model_validate(json.loads(candidate))
                except (ValidationError, json.JSONDecodeError) as exc2:
                    last_error = exc2

        raise ParseError(
            f"Failed to parse output into {model.__name__}: {last_error}"
        )

    def parse_or_none(self, text: str, model: Type[T]) -> T | None:
        """Like :meth:`parse` but returns ``None`` on failure."""
        try:
            return self.parse(text, model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Structured parse failed (returning None): %s", exc)
            return None

    # -- retry-aware helper -----------------------------------------------

    async def parse_with_retry(
        self,
        regenerate,
        model: Type[T],
        *,
        prompt: str,
        system: str | None = None,
        max_retries: int | None = None,
    ) -> T:
        """Call ``regenerate`` to get text, parse it, and retry on failure.

        ``regenerate`` is an async callable ``(prompt: str) -> str`` that
        produces fresh model output (e.g. re-invoking the LLM with a repair
        hint appended to the prompt).
        """
        retries = self.max_retries if max_retries is None else max_retries
        current_prompt = prompt
        last_error: Exception | None = None

        for attempt in range(retries + 1):
            text = await regenerate(current_prompt)
            try:
                return self.parse(text, model)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < retries:
                    logger.warning(
                        "Parse attempt %d/%d failed: %s; retrying with repaired prompt",
                        attempt + 1,
                        retries + 1,
                        exc,
                    )
                    current_prompt = (
                        f"{current_prompt}\n\n"
                        "Your previous response failed validation. "
                        f"Error: {exc}\n"
                        "Return ONLY valid JSON matching the requested schema. "
                        "Do not include reasoning tags or markdown fences."
                    )

        raise ParseError(
            f"Exhausted {retries + 1} parse attempts for {model.__name__}: {last_error}"
        )
