from __future__ import annotations

from abc import ABC, abstractmethod


class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, temperature: float | None = None) -> str:
        raise NotImplementedError()
