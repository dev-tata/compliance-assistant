from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas.deliverables import DeliverableExtractionRequest, DeliverableExtractionResponse, DeliverableExtractionSummary
from app.services.case_service import (
    delete_case_deliverable_result,
    get_case_compliance_payload,
    get_case_deliverable_result,
    list_case_deliverables,
)
from app.services.deliverable_extraction_service import run_case_deliverable_extraction
from app.services.llm.errors import (
    LLMConfigurationError,
    LLMGenerationError,
    LLMQuotaExceededError,
)

router = APIRouter(prefix="/cases", tags=["deliverables"])


@router.get("/{case_id}/deliverables", response_model=list[DeliverableExtractionSummary])
def get_case_deliverable_runs(case_id: str):
    return list_case_deliverables(case_id)


@router.get("/{case_id}/deliverables/{file_name}", response_model=DeliverableExtractionResponse)
def get_case_deliverable_run(case_id: str, file_name: str):
    return get_case_deliverable_result(case_id, file_name)


@router.delete("/{case_id}/deliverables/{file_name}", response_model=DeliverableExtractionSummary)
def delete_case_deliverable_run(case_id: str, file_name: str):
    return delete_case_deliverable_result(case_id, file_name)


@router.post("/{case_id}/deliverables/extract", response_model=DeliverableExtractionResponse)
def extract_case_deliverables(case_id: str, request: DeliverableExtractionRequest):
    case_payload = get_case_compliance_payload(case_id)
    if not case_payload["procedures"]:
        raise HTTPException(
            status_code=400,
            detail="Case must contain at least one procedure document for deliverable extraction.",
        )

    try:
        return run_case_deliverable_extraction(
            case_id=case_id,
            case_payload=case_payload,
            request=request,
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LLMQuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except LLMGenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
