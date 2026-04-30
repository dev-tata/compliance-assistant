from __future__ import annotations

from google import genai

from app.services.llm.base import BaseLLM
from app.services.llm.config import get_required_env, resolve_generation_temperature
from app.services.llm.errors import LLMGenerationError, LLMQuotaExceededError


class GeminiService(BaseLLM):
    def __init__(self, model: str) -> None:
        self.provider = "gemini"
        self.client = genai.Client(api_key=get_required_env("GEMINI_API_KEY"))
        self.model = model

    def generate(self, prompt: str, *, temperature: float | None = None) -> str:
        resolved_temperature = resolve_generation_temperature(
            requested_temperature=temperature,
            provider_default=0.2,
        )
        self._log_generate_start(prompt, temperature=resolved_temperature)
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": resolved_temperature,
                    "response_mime_type": "application/json",
                },
            )
        except Exception as exc:
            message = str(exc)
            self._log_generate_error(message)
            normalized = message.lower()
            if _is_quota_exceeded_error(normalized):
                raise LLMQuotaExceededError(_build_quota_exceeded_message(message)) from exc
            raise LLMGenerationError(f"Gemini generation failed: {message}") from exc

        response_text = response.text or ""
        self._log_generate_success(response_text)
        return response_text


def _is_quota_exceeded_error(message: str) -> bool:
    return (
        "resource_exhausted" in message
        or "monthly spending cap" in message
        or "spend cap" in message
        or ("429" in message and "quota" in message)
    )


def _build_quota_exceeded_message(message: str) -> str:
    if "monthly spending cap" in message.lower() or "spend cap" in message.lower():
        return (
            "Gemini generation failed: project spend cap exceeded. "
            "Increase the cap at https://ai.studio/spend or switch provider/model."
        )
    return "Gemini generation failed: provider quota exceeded. Retry later or switch provider/model."
