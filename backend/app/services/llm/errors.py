from __future__ import annotations


class LLMError(Exception):
    """Base class for provider-layer LLM errors."""


class LLMConfigurationError(LLMError):
    """Raised when provider configuration is invalid or incomplete."""


class LLMGenerationError(LLMError):
    """Raised when a provider call fails during text generation."""
