from __future__ import annotations

from openai import OpenAI

from app.services.llm.base import BaseLLM
from app.services.llm.config import DEFAULT_SYSTEM_PROMPT, get_required_env, resolve_generation_temperature
from app.services.llm.errors import LLMGenerationError, LLMQuotaExceededError


class OpenAIService(BaseLLM):
    def __init__(self, model: str) -> None:
        self.provider = "openai"
        self.model = model
        self.client = OpenAI(api_key=get_required_env("OPENAI_API_KEY"))

    def generate(self, prompt: str, *, temperature: float | None = None) -> str:
        resolved_temperature = resolve_generation_temperature(
            requested_temperature=temperature,
            provider_default=None,
        )
        self._log_generate_start(prompt, temperature=resolved_temperature)
        try:
            request_kwargs = {
                "model": self.model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            }
            if resolved_temperature is not None and resolved_temperature != 0.0:
                request_kwargs["temperature"] = resolved_temperature

            response = self.client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            message = str(exc)
            self._log_generate_error(message)
            normalized = message.lower()
            if _is_quota_exceeded_error(normalized):
                raise LLMQuotaExceededError(
                    "OpenAI generation failed: provider quota exceeded. Check billing/limits or switch provider/model."
                ) from exc
            raise LLMGenerationError(f"OpenAI generation failed: {message}") from exc

        response_text = response.choices[0].message.content or ""
        self._log_generate_success(response_text)
        return response_text


def _is_quota_exceeded_error(message: str) -> bool:
    return (
        "insufficient_quota" in message
        or ("429" in message and "quota" in message)
        or "rate limit" in message
        or "rate_limit" in message
    )
