from __future__ import annotations

from openai import OpenAI

from app.services.llm.base import BaseLLM
from app.services.llm.config import DEFAULT_SYSTEM_PROMPT, get_required_env
from app.services.llm.errors import LLMGenerationError


class OpenAIService(BaseLLM):
    def __init__(self, model: str) -> None:
        self.model = model
        self.client = OpenAI(api_key=get_required_env("OPENAI_API_KEY"))

    def generate(self, prompt: str, *, temperature: float | None = None) -> str:
        try:
            request_kwargs = {
                "model": self.model,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            }
            if temperature is not None and temperature != 0.0:
                request_kwargs["temperature"] = temperature

            response = self.client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            raise LLMGenerationError(f"OpenAI generation failed: {exc}") from exc

        return response.choices[0].message.content or ""
