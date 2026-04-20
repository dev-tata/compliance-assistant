from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from app.schemas.compliance import (
    ComplianceAnalysis,
    ComplianceLinkedRow,
    ComplianceMethod,
    ComplianceRequest,
    ComplianceResponse,
)
from app.services.compliance_scoring_service import compute_scores, enrich_analysis_for_scoring
from app.services.document_service import COMPLIANCE_DIR, current_timestamp
from app.services.llm.errors import LLMGenerationError
from app.services.llm.factory import get_llm_service
from app.services.llm.json_utils import extract_json_object


def execute_compliance_method(
    *,
    case_id: str,
    case_payload: dict[str, object],
    request: ComplianceRequest,
    method: ComplianceMethod,
    prompt: str,
) -> ComplianceResponse:
    llm_service = get_llm_service(request.provider, request.model)
    raw_analysis = llm_service.generate(prompt)
    try:
        analysis = ComplianceAnalysis(**extract_json_object(raw_analysis))
    except (ValidationError, ValueError) as exc:
        raise LLMGenerationError(f"Invalid compliance response from model: {exc}") from exc

    if analysis.procedure_to_record and not analysis.findings:
        analysis = analysis.model_copy(update={"findings": analysis.procedure_to_record})
    if not analysis.linked_rows and analysis.procedure_to_record:
        linked_rows = _build_linked_rows(analysis)
        analysis = analysis.model_copy(update={"linked_rows": linked_rows})
    elif analysis.linked_rows and analysis.procedure_to_record:
        linked_rows = _resolve_linked_row_requirements(analysis)
        analysis = analysis.model_copy(update={"linked_rows": linked_rows})

    analysis = enrich_analysis_for_scoring(analysis)
    scores = compute_scores(analysis)

    saved_path = build_compliance_result_path(case_id)
    created_at = current_timestamp()
    response = ComplianceResponse(
        case_id=case_id,
        compliance_provider=request.provider,
        compliance_model=request.model,
        extraction_provider=case_payload.get("extraction_provider") if isinstance(case_payload, dict) else None,
        extraction_model=case_payload.get("extraction_model") if isinstance(case_payload, dict) else None,
        method=method,
        created_at=created_at,
        saved_at=saved_path.as_posix(),
        analysis=analysis,
        scores=scores,
        section_matches=[],
    )
    saved_path.write_text(
        response.model_dump_json(
            indent=2,
            exclude_none=True,
        ),
        encoding="utf-8",
    )
    return response


def build_shared_output_instructions(
    *,
    include_record_to_procedure: bool = False,
) -> list[str]:
    output_structure = [
        '{'
        '"overall_assessment":"partial|satisfied|not_satisfied or short text",'
        '"linked_rows":[{"requirement_ref":"REQ-1","status":"satisfied|partial|not_satisfied","gap":"...","recommendation":"..."}],'
        '"procedure_to_record":[{"requirement":"...","status":"satisfied|partial|not_satisfied","evidence":["..."],"source_documents":["record-file-name"]}],'
    ]
    output_structure.append('"recommended_actions":["..."]' '}')

    directional_instructions = [
        "- Provide one directional view:",
        "  1. procedure_to_record: evaluate each required procedure item against the record documents.",
        "- Do not include record_to_procedure in the response.",
    ]

    decision_policy = [
        "- satisfied: explicit and adequate supporting evidence exists in the record documents",
        "- partial: some support exists, but it is incomplete, indirect, weak, or ambiguous",
        "- not_satisfied: no meaningful supporting evidence exists in the record documents",
    ]
    return [
        "You are a compliance analysis engine.",
        "Your task is to evaluate whether the record documents satisfy the explicit requirements stated in the procedure documents.",
        "Definitions:",
        "- procedure documents: the normative source of requirements",
        "- extracted deliverables: normalized requirements previously extracted from procedure documents",
        "- record documents: the only allowed source of compliance evidence",
        "Hard constraints:",
        "- Use only the documents included in this prompt.",
        "- Extract requirements only from procedure documents or from extracted deliverables provided from those procedure documents.",
        "- Use evidence only from record documents.", 
        "- Never use procedure documents as proof that compliance is satisfied.",
        "- Never cite a procedure document in source_documents.",
        "- Do not invent requirements, evidence, sections, systems, roles, or context.",
        "- If evidence is missing, ambiguous, indirect, or incomplete, use partial or not_satisfied.",
        "- Be conservative. Do not assume compliance.",
        "Output requirements:",
        "- Return one JSON object only.",
        "- Do not return markdown.",
        "- Do not wrap the JSON in code fences.",
        "- Do not include commentary before or after the JSON.",
        "- Every procedure_to_record item and linked_rows item must use one of these status values only: satisfied, partial, not_satisfied.",
        "- Every procedure_to_record.source_documents item must refer only to record document filenames from this case.",
        "- Every procedure_to_record.evidence item must be a short plain-text statement grounded in the record documents.",
        "- Include one procedure_to_record item per distinct requirement or merged closely related requirement.",
        "- Include one linked_rows item per procedure_to_record item.",
        "- In linked_rows, use requirement_ref instead of repeating the full requirement text.",
        "- requirement_ref must be a short stable identifier such as REQ-1, REQ-2, etc., matching the corresponding procedure_to_record item order.",
        "- In linked_rows, gap should briefly state the missing or weak point for partial/not_satisfied items, and may be empty for satisfied items.",
        "- In linked_rows, recommendation should briefly state the corrective action for partial/not_satisfied items, and may be empty for satisfied items.",
        *directional_instructions,
        "- Do not include findings in the model output. The backend will derive findings from procedure_to_record.",
        "Return JSON with exactly this structure:",
        *output_structure,
        "Decision policy:",
        *decision_policy,
    ]


def build_compliance_result_path(case_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"case_{case_id}_compliance_{timestamp}_{uuid4().hex}.json"
    return COMPLIANCE_DIR / filename


def _build_linked_rows(analysis: ComplianceAnalysis) -> list[ComplianceLinkedRow]:
    gaps = list(analysis.gaps)
    recommendations = list(analysis.recommended_actions)
    linked_rows: list[ComplianceLinkedRow] = []
    for index, finding in enumerate(analysis.procedure_to_record):
        gap = ""
        recommendation = ""
        if finding.status != "satisfied":
            gap = gaps.pop(0) if gaps else ""
            recommendation = recommendations.pop(0) if recommendations else ""
        linked_rows.append(
            ComplianceLinkedRow(
                requirement_ref=f"REQ-{index + 1}",
                requirement=finding.requirement,
                status=finding.status,
                gap=gap,
                recommendation=recommendation,
            )
        )
    return linked_rows


def _resolve_linked_row_requirements(analysis: ComplianceAnalysis) -> list[ComplianceLinkedRow]:
    requirement_map = {
        f"REQ-{index + 1}": finding.requirement
        for index, finding in enumerate(analysis.procedure_to_record)
    }
    resolved_rows: list[ComplianceLinkedRow] = []
    for row in analysis.linked_rows:
        requirement = row.requirement
        if not requirement and row.requirement_ref:
            requirement = requirement_map.get(row.requirement_ref, "")
        resolved_rows.append(
            row.model_copy(
                update={
                    "requirement": requirement,
                }
            )
        )
    return resolved_rows
