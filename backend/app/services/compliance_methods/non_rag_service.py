from __future__ import annotations

from typing import Any

from app.schemas.compliance import ComplianceAnalysis, ComplianceRequest
from app.services.compliance_methods.compliance_method_common import (
    build_compliance_prompt,
    extract_allowed_record_documents,
    execute_compliance_method,
    parse_compliance_analysis_response,
    simplify_document_for_prompt,
)
from app.services.llm.errors import LLMGenerationError


def run_non_rag_compliance(
    *,
    case_id: str,
    case_payload: dict[str, Any],
    request: ComplianceRequest,
):
    return execute_compliance_method(
        case_id=case_id,
        case_payload=case_payload,
        request=request,
        method="non_rag",
        prompt=build_non_rag_prompt(case_payload=case_payload, instructions=request.instructions),
    )


def _build_non_rag_case_context(case_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "method": "non_rag",
        "requirement_source": "deliverables",
        "case_id": case_payload.get("case_id"),
        "title": case_payload.get("title"),
        "notes": case_payload.get("notes"),
        "deliverables": case_payload.get("deliverables", []),
        "records": [
            simplify_document_for_prompt(document) for document in case_payload.get("records", [])
        ],
    }


def build_non_rag_prompt(
    *,
    case_payload: dict[str, Any],
    instructions: str | None,
) -> str:
    return build_compliance_prompt(
        method="non_rag",
        payload=_build_non_rag_case_context(case_payload),
        instructions=instructions,
    )


def evaluate_non_rag_analysis(
    *,
    llm_service: Any,
    case_payload: dict[str, Any],
    instructions: str | None,
) -> ComplianceAnalysis:
    prompt = build_non_rag_prompt(case_payload=case_payload, instructions=instructions)
    try:
        return parse_compliance_analysis_response(
            raw_analysis=llm_service.generate(prompt, temperature=0.0),
            method="non_rag",
            expected_count=len(case_payload.get("deliverables", [])),
            allowed_record_documents=extract_allowed_record_documents(case_payload),
        )
    except LLMGenerationError as exc:
        raise LLMGenerationError(f"Invalid non_rag baseline response: {exc}") from exc
