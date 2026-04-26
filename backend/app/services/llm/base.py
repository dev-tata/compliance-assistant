from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone


class BaseLLM(ABC):
    provider: str
    model: str

    @abstractmethod
    def generate(self, prompt: str, *, temperature: float | None = None) -> str:
        raise NotImplementedError()

    def _log_generate_start(self, prompt: str, *, temperature: float | None = None) -> None:
        self._log_generate_event(
            event="start",
            prompt_length=len(prompt),
            temperature=temperature,
        )

    def _log_generate_success(self, response_text: str) -> None:
        self._log_generate_event(
            event="success",
            response_length=len(response_text),
        )

    def _log_generate_error(self, message: str) -> None:
        self._log_generate_event(
            event="error",
            error=message,
        )

    def _log_generate_event(self, *, event: str, **details: object) -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        detail_text = " ".join(
            f"{key}={value!r}" for key, value in details.items() if value is not None
        )
        print(
            f"[LLM] ts={timestamp} provider={getattr(self, 'provider', 'unknown')} "
            f"model={getattr(self, 'model', 'unknown')} event={event}"
            + (f" {detail_text}" if detail_text else ""),
            flush=True,
        )
