from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv


load_dotenv()


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    try:
        result = int(value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be an integer; received {value!r}"
        ) from exc

    return result


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)

    if value is None or not value.strip():
        return default

    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a number; received {value!r}"
        ) from exc

    if not math.isfinite(result):
        raise ValueError(
            f"{name} must be a finite number; received {value!r}"
        )

    return result


@dataclass(frozen=True, slots=True)
class Settings:
    llm_model: str
    llm_api_key: str | None
    llm_api_base: str | None
    llm_temperature: float
    llm_timeout: float
    agent_max_steps: int
    agent_max_context_chars: int
    tool_timeout: float
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        model = os.getenv(
            "LLM_MODEL",
            "ollama_chat/qwen2.5:3b",
        ).strip()

        if not model:
            raise ValueError("LLM_MODEL cannot be empty")

        temperature = _get_float("LLM_TEMPERATURE", 0.0)
        llm_timeout = _get_float("LLM_TIMEOUT", 60.0)
        tool_timeout = _get_float("TOOL_TIMEOUT", 30.0)
        max_steps = _get_int("AGENT_MAX_STEPS", 10)
        max_context_chars = _get_int("AGENT_MAX_CONTEXT_CHARS", 120_000)

        if max_context_chars < 1:
            raise ValueError(
                "AGENT_MAX_CONTEXT_CHARS must be at least 1"
            )

        if not 0.0 <= temperature <= 2.0:
            raise ValueError("LLM_TEMPERATURE must be between 0 and 2")

        if llm_timeout <= 0:
            raise ValueError("LLM_TIMEOUT must be greater than zero")

        if tool_timeout <= 0:
            raise ValueError("TOOL_TIMEOUT must be greater than zero")

        if max_steps < 1:
            raise ValueError("AGENT_MAX_STEPS must be at least 1")

        api_key = (
            os.getenv("LLM_API_KEY")
            or os.getenv("XAI_1_API_KEY")
            or None
        )

        api_base = (
            os.getenv("LLM_API_BASE")
            or os.getenv("OPENAI_API_BASE")
            or None
        )

        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        valid_levels = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"}

        if log_level not in valid_levels:
            raise ValueError(
                f"LOG_LEVEL must be one of {sorted(valid_levels)}"
            )

        return cls(
            llm_model=model,
            llm_api_key=api_key,
            llm_api_base=api_base,
            llm_temperature=temperature,
            llm_timeout=llm_timeout,
            agent_max_steps=max_steps,
            agent_max_context_chars=max_context_chars,
            tool_timeout=tool_timeout,
            log_level=log_level,
        )


def require_api_key(settings: Settings) -> None:
    """Fail early with a clear message when no provider API key is configured."""
    if not settings.llm_api_key:
        raise RuntimeError(
            "No API key found. Set LLM_API_KEY or XAI_1_API_KEY "
            "in the project .env file."
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process settings, validated and cached on first use.

    Reading the environment at import time made the module impossible to
    import without a fully valid configuration (and made tests depend on
    whatever was in the developer's ``.env``). Resolve it lazily instead.
    """
    return Settings.from_env()
