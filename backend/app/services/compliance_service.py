from __future__ import annotations

from typing import Any

from app.schemas.compliance import ComplianceRequest, ComplianceResponse
from app.services.compliance_methods.multi_source_rag_service import run_multi_source_rag_compliance
from app.services.compliance_methods.non_rag_service import run_non_rag_compliance
from app.services.compliance_methods.single_source_rag_service import run_single_source_rag_compliance


def run_case_compliance_analysis(
    *,
    case_id: str,
    case_payload: dict[str, Any],
    request: ComplianceRequest,
) -> ComplianceResponse:
    if request.method == "single_source_rag":
        return run_single_source_rag_compliance(
            case_id=case_id,
            case_payload=case_payload,
            request=request,
        )

    if request.method == "multi_source_rag":
        return run_multi_source_rag_compliance(
            case_id=case_id,
            case_payload=case_payload,
            request=request,
        )

    return run_non_rag_compliance(
        case_id=case_id,
        case_payload=case_payload,
        request=request,
    )
