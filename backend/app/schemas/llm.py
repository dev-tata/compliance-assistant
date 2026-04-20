from __future__ import annotations

from pydantic import BaseModel


class LLMProviderDescriptor(BaseModel):
    key: str
    label: str
    default_model: str
    description: str
    endpoint_mode: str
