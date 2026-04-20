from __future__ import annotations

from fastapi import APIRouter

from app.schemas.llm import LLMProviderDescriptor
from app.services.llm.provider_catalog import list_provider_catalog

router = APIRouter(prefix="/llm", tags=["llm"])


@router.get("/providers", response_model=list[LLMProviderDescriptor])
def get_llm_providers():
    return [LLMProviderDescriptor(**provider.__dict__) for provider in list_provider_catalog()]
