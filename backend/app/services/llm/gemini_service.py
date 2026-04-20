from __future__ import annotations

from google import genai

from app.services.llm.base import BaseLLM
from app.services.llm.config import get_required_env
from app.services.llm.errors import LLMGenerationError


class GeminiService(BaseLLM):
    def __init__(self, model: str) -> None:
        self.client = genai.Client(api_key=get_required_env("GEMINI_API_KEY"))
        self.model = model

    def generate(self, prompt: str, *, temperature: float | None = None) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "temperature": temperature if temperature is not None else 0.2,
                    "response_mime_type": "application/json",
                },
            )
        except Exception as exc:
            raise LLMGenerationError(f"Gemini generation failed: {exc}") from exc

        return response.text or ""
