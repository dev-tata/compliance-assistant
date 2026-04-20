from __future__ import annotations

from typing import Type

from app.services.llm.base import BaseLLM
from app.services.llm.errors import LLMConfigurationError
from app.services.llm.gemini_service import GeminiService
from app.services.llm.openai_service import OpenAIService
from app.services.llm.provider_catalog import PROVIDER_CATALOG

PROVIDER_REGISTRY: dict[str, Type[BaseLLM]] = {
    "gemini": GeminiService,
    "openai": OpenAIService,
}


def get_llm_service(provider: str, model: str) -> BaseLLM:
    provider_key = provider.lower()
    service_class = PROVIDER_REGISTRY.get(provider_key)
    if not service_class:
        supported = ", ".join(sorted(PROVIDER_CATALOG.keys()))
        raise LLMConfigurationError(
            f"Unsupported provider: {provider_key}. Supported providers: {supported}"
        )
    return service_class(model=model)
