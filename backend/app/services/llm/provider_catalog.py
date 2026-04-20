from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMProviderMetadata:
    key: str
    label: str
    default_model: str
    description: str
    endpoint_mode: str


PROVIDER_CATALOG: dict[str, LLMProviderMetadata] = {
    "openai": LLMProviderMetadata(
        key="openai",
        label="OpenAI",
        default_model="gpt-5.4-nano",
        description="Hosted OpenAI chat completion models.",
        endpoint_mode="remote",
    ),
    "gemini": LLMProviderMetadata(
        key="gemini",
        label="Gemini",
        default_model="gemini-3.1-flash-lite-preview",
        description="Hosted Google Gemini models.",
        endpoint_mode="remote",
    ),
}


def list_provider_catalog() -> list[LLMProviderMetadata]:
    return [PROVIDER_CATALOG[key] for key in sorted(PROVIDER_CATALOG.keys())]
