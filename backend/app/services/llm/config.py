from __future__ import annotations

import os

from dotenv import load_dotenv

from app.services.llm.errors import LLMConfigurationError

load_dotenv()

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
