from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from app.services.llm.errors import LLMConfigurationError
from app.services.runtime_config import is_evaluation_v3_experiment_mode

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

DEFAULT_SYSTEM_PROMPT = (
    "You are a careful compliance analyst. "
    "Return valid JSON only when the user asks for structured output."
)


def get_env(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    if value is None or value == "":
        return default
    return value


def get_optional_env(key: str, default: str | None = None) -> str:
    return get_env(key, default) or ""


def get_required_env(key: str) -> str:
    value = get_env(key)
    if not value:
        raise LLMConfigurationError(f"{key} is not set")
    return value


def resolve_generation_temperature(
    *,
    requested_temperature: float | None,
    provider_default: float | None = None,
) -> float | None:
    if is_evaluation_v3_experiment_mode():
        return 0.0
    if requested_temperature is not None:
        return requested_temperature
    return provider_default
