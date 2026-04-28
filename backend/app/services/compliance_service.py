from __future__ import annotations

from typing import Any

from app.schemas.compliance import ComplianceRequest, ComplianceResponse
from app.services.compliance_methods.two_stage_rag_service import run_two_stage_rag_compliance


def run_case_compliance_analysis(
    *,
    case_id: str,
    case_payload: dict[str, Any],
    request: ComplianceRequest,
) -> ComplianceResponse:
    return run_two_stage_rag_compliance(
        case_id=case_id,
        case_payload=case_payload,
        request=request,
    )
