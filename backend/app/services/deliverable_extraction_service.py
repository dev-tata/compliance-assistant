from __future__ import annotations

from typing import Any

from app.schemas.deliverables import DeliverableExtractionRequest, DeliverableExtractionResponse
from app.services.deliverable_methods.extraction_method import (
    run_non_rag_deliverable_extraction,
)


def run_case_deliverable_extraction(
    *,
    case_id: str,
    case_payload: dict[str, Any],
    request: DeliverableExtractionRequest,
) -> DeliverableExtractionResponse:
    if request.method == "non_rag":
        return run_non_rag_deliverable_extraction(
            scope_id=case_id,
            case_payload=case_payload,
            request=request,
            case_id=case_id,
        )

    if request.method in {"simple_rag", "nested_rag"}:
        raise NotImplementedError(f"Deliverable extraction method '{request.method}' is not implemented yet")

    raise NotImplementedError(f"Unsupported deliverable extraction method '{request.method}'")


def run_document_deliverable_extraction(
    *,
    stored_filename: str,
    document_payload: dict[str, Any],
    request: DeliverableExtractionRequest,
) -> DeliverableExtractionResponse:
    if request.method == "non_rag":
        return run_non_rag_deliverable_extraction(
            scope_id=stored_filename,
            case_payload=document_payload,
            request=request,
            document_stored_filename=stored_filename,
            source_filename=document_payload.get("procedures", [{}])[0].get("source_filename"),
        )

    if request.method in {"simple_rag", "nested_rag"}:
        raise NotImplementedError(f"Deliverable extraction method '{request.method}' is not implemented yet")

    raise NotImplementedError(f"Unsupported deliverable extraction method '{request.method}'")
