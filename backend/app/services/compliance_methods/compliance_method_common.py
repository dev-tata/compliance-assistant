from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from app.schemas.compliance import (
    ComplianceAnalysis,
    ComplianceFinding,
    ComplianceLinkedRow,
    ComplianceMethod,
    ComplianceRequest,
    ComplianceResponse,
)
from app.services.compliance_scoring_service import compute_scores, enrich_analysis_for_scoring
from app.services.document_service import current_timestamp
from app.services.llm.errors import LLMGenerationError
from app.services.llm.factory import get_llm_service
from app.services.llm.json_utils import extract_json_object
from app.services.retrieval.faiss_retrieval import normalize_whitespace
from app.services.storage_paths import get_case_compliance_dir

PROMPT_FAMILY = "canonical_compliance_v1"
STATUS_COMPLETION_SCORES = {
    "satisfied": 100,
    "partial": 50,
    "not_satisfied": 0,
}


def execute_compliance_method(
    *,
    case_id: str,
    case_payload: dict[str, object],
    request: ComplianceRequest,
    method: ComplianceMethod,
    prompt: str,
) -> ComplianceResponse:
    llm_service = get_llm_service(request.provider, request.model)
    allowed_record_documents = extract_allowed_record_documents(case_payload)
    analysis = parse_compliance_analysis_response(
        raw_analysis=llm_service.generate(prompt, temperature=0.0),
        method=method,
        expected_count=len(case_payload.get("deliverables", [])) if isinstance(case_payload, dict) else 0,
        allowed_record_documents=allowed_record_documents,
    )
    analysis = enrich_analysis_for_scoring(analysis)
    analysis = apply_computed_overall_assessment(analysis)
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
        reference_stored_filenames=[
            item.get("stored_filename")
            for item in case_payload.get("references", [])
            if isinstance(item, dict) and item.get("stored_filename")
        ] if isinstance(case_payload, dict) else [],
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


def build_shared_output_instructions(*, single_requirement: bool = False) -> list[str]:
    output_structure = [
        "{",
        '"overall_assessment":"computed_by_backend",',
        '"linked_rows":[{"requirement_ref":"REQ-1","status":"satisfied|partial|not_satisfied","gap":"...","recommendation":"..."}],',
        '"procedure_to_record":[{"requirement":"...","status":"satisfied|partial|not_satisfied","evidence":["verbatim or near-verbatim short quotes"],"source_documents":["record-file-name"]}],',
        '"recommended_actions":["..."]',
        "}",
    ]
    return [
        "You are a compliance adjudicator.",
        "Evaluate whether the record documents satisfy the explicit requirements stated in the extracted procedure deliverables.",
        "Task rules:",
        "- Evaluate each requirement independently.",
        "- Use only admissible evidence from the input payload for this method.",
        "- Treat extracted deliverables as the canonical requirement source.",
        "- Provide evidence as short quote-based plain text grounded in admissible record evidence.",
        "- Do not invent requirements, evidence, sections, systems, roles, or context.",
        "Status definitions:",
        "- satisfied: all required elements are explicitly supported by admissible record evidence.",
        "- partial: some required elements are supported, but at least one required element is missing.",
        "- not_satisfied: required support is absent, ambiguous, contradictory, indirect, or not admissible.",
        "Predicate-level decision logic:",
        "- Break each requirement into the concrete elements it requires.",
        "- Check each element against admissible evidence only.",
        "- If all elements are supported, return satisfied.",
        "- If some elements are supported but one or more are missing, return partial.",
        "- If support is ambiguous, conflicting, too weak, or missing entirely, return not_satisfied.",
        "- If support is indirect but still admissible and sufficient to establish the required element, you may treat it as satisfied or partial depending on completeness.",
        "Output requirements:",
        "- Return one JSON object only.",
        "- Do not return markdown or code fences.",
        "- The backend computes overall_assessment and completion_percent. Set overall_assessment to computed_by_backend.",
        (
            "- Include exactly one procedure_to_record item and one linked_rows item for the single input requirement."
            if single_requirement
            else "- Include exactly one procedure_to_record item per input requirement in the same order."
        ),
        (
            "- Use requirement_ref REQ-1 for the single input requirement."
            if single_requirement
            else "- Include one linked_rows item per procedure_to_record item in the same order."
        ),
        (
            "- Do not emit any requirement_ref other than REQ-1."
            if single_requirement
            else "- Use requirement_ref values REQ-1, REQ-2, and so on, matching the input order."
        ),
        "- Every procedure_to_record item and linked_rows item must use one of these status values only: satisfied, partial, not_satisfied.",
        "- Every procedure_to_record.source_documents item must refer only to record document filenames from this case.",
        "- For satisfied items, linked_rows.gap and linked_rows.recommendation may be empty.",
        "- For partial and not_satisfied items, gap must briefly state the missing point and recommendation must briefly state the corrective action.",
        "- Do not include findings in the model output. The backend will derive findings from procedure_to_record.",
        "Return JSON with exactly this structure:",
        *output_structure,
    ]


def build_method_specific_constraints(method: ComplianceMethod) -> list[str]:
    shared = [f"Method metadata: {method}."]
    if method == "non_rag":
        return [
            *shared,
            "- Use full records as admissible evidence.",
            "- You may search anywhere inside records.",
            "- Do not use retrieval results because none are provided for this method.",
        ]
    if method == "single_source_rag":
        return [
            *shared,
            "- Use ONLY retrieved_record_sections as admissible evidence.",
            "- Do not rely on any full record document outside retrieved_record_sections.",
            "- If a claim is supported only outside retrieved_record_sections, treat it as not admissible and return partial or not_satisfied accordingly.",
        ]
    return [
        *shared,
        "- Use ONLY retrieved_record_sections as admissible compliance evidence.",
        "- Use retrieved_requirement_context only to interpret the meaning of a requirement.",
        "- Never use retrieved_requirement_context as compliance evidence and never cite reference documents in source_documents.",
        "- Do not rely on any full record document outside retrieved_record_sections.",
        "- If a claim is supported only outside retrieved_record_sections, treat it as not admissible and return partial or not_satisfied accordingly.",
    ]


def build_compliance_prompt(
    *,
    method: ComplianceMethod,
    payload: dict[str, Any],
    instructions: str | None,
    single_requirement: bool = False,
) -> str:
    prompt_parts = [
        *build_shared_output_instructions(single_requirement=single_requirement),
        "Input payload notes:",
        "- The payload contains the requirement list plus method-specific context blocks.",
        *build_method_specific_constraints(method),
        f"Prompt family: {PROMPT_FAMILY}",
        f"Compliance payload:\n{json.dumps(payload, ensure_ascii=False, indent=2)}",
    ]
    if instructions:
        prompt_parts.append(f"Additional user instructions:\n{instructions}")
    return "\n\n".join(prompt_parts)


def serialize_baseline_analysis(analysis: ComplianceAnalysis) -> list[dict[str, Any]]:
    findings = analysis.procedure_to_record or analysis.findings
    return [
        {
            "requirement_ref": f"REQ-{index + 1}",
            "requirement": finding.requirement,
            "status": finding.status,
            "evidence": finding.evidence,
            "source_documents": finding.source_documents,
        }
        for index, finding in enumerate(findings)
    ]


def extract_allowed_record_documents(case_payload: dict[str, Any] | dict[str, object]) -> set[str]:
    if not isinstance(case_payload, dict):
        return set()
    return {
        normalize_whitespace(document.get("source_filename") or document.get("stored_filename"))
        for document in case_payload.get("records", [])
        if isinstance(document, dict)
        and normalize_whitespace(document.get("source_filename") or document.get("stored_filename"))
    }


def simplify_document_for_prompt(document: dict[str, Any]) -> dict[str, Any]:
    parsed_json = document.get("parsed_json") or {}
    return {
        "document_type": document.get("document_type"),
        "source_filename": document.get("source_filename"),
        "stored_filename": document.get("stored_filename"),
        "group_id": document.get("group_id"),
        "language": document.get("language"),
        "parsed_json": {
            "source_filename": parsed_json.get("source_filename"),
            "stored_filename": parsed_json.get("stored_filename"),
            "parser_used": parsed_json.get("parser_used"),
            "metadata": parsed_json.get("metadata"),
            "sections": parsed_json.get("sections", []),
        },
    }


def build_compliance_result_path(case_id: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"case_{case_id}_compliance_{timestamp}_{uuid4().hex}.json"
    return get_case_compliance_dir(case_id) / filename


def build_requirement_query_text(deliverable: dict[str, Any]) -> str:
    return " ".join(
        part
        for part in (
            deliverable.get("requirement_text", ""),
            deliverable.get("source_quote", ""),
            deliverable.get("heading_title", ""),
        )
        if normalize_whitespace(part)
    )


def serialize_retrieved_section(chunk: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_document": chunk.get("source_document"),
        "section_label": chunk.get("section_label"),
        "heading_title": chunk.get("heading_title"),
        "text": chunk.get("text"),
        "table_markdown": chunk.get("table_markdown"),
        "faiss_score": chunk.get("faiss_score"),
        "reranker_score": chunk.get("reranker_score"),
        "retrieval_score": chunk.get("retrieval_score"),
    }


def serialize_deliverable_for_prompt(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_document": item.get("source_document"),
        "section_label": item.get("section_label"),
        "heading_title": item.get("heading_title"),
        "requirement_text": item.get("requirement_text"),
        "source_quote": item.get("source_quote"),
        "retrieval_score": item.get("retrieval_score"),
    }


def build_single_requirement_prompt(
    *,
    method: ComplianceMethod,
    requirement_payload: dict[str, Any],
    instructions: str | None,
) -> str:
    return build_compliance_prompt(
        method=method,
        payload=requirement_payload,
        instructions=instructions,
        single_requirement=True,
    )


def parse_compliance_analysis_response(
    *,
    raw_analysis: str,
    method: str,
    expected_count: int,
    allowed_record_documents: set[str] | None = None,
) -> ComplianceAnalysis:
    try:
        analysis = ComplianceAnalysis(**extract_json_object(raw_analysis))
    except (ValidationError, ValueError) as exc:
        raise LLMGenerationError(f"Invalid compliance response from model: {exc}") from exc
    return normalize_compliance_analysis(
        analysis=analysis,
        method=method,
        expected_count=expected_count,
        allowed_record_documents=allowed_record_documents,
    )


def normalize_compliance_analysis(
    *,
    analysis: ComplianceAnalysis,
    method: str,
    expected_count: int,
    allowed_record_documents: set[str] | None = None,
) -> ComplianceAnalysis:
    validate_analysis_cardinality(
        analysis=analysis,
        expected_count=expected_count,
        method=method,
    )
    if analysis.procedure_to_record and not analysis.findings:
        analysis = analysis.model_copy(update={"findings": analysis.procedure_to_record})
    if allowed_record_documents is not None:
        analysis = _normalize_analysis_source_documents(
            analysis=analysis,
            allowed_record_documents=allowed_record_documents,
        )
    if analysis.linked_rows:
        analysis = analysis.model_copy(
            update={
                "linked_rows": [
                    _sanitize_linked_row_recommendation(row)
                    for row in analysis.linked_rows
                ]
            }
        )
    if not analysis.linked_rows and analysis.procedure_to_record:
        analysis = analysis.model_copy(update={"linked_rows": _build_linked_rows(analysis)})
    elif analysis.linked_rows and analysis.procedure_to_record:
        analysis = analysis.model_copy(update={"linked_rows": _resolve_linked_row_requirements(analysis)})
    return analysis


def evaluate_single_requirement(
    *,
    llm_service: Any,
    method: ComplianceMethod,
    requirement_payload: dict[str, Any],
    instructions: str | None,
    temperature: float = 0.0,
) -> ComplianceAnalysis:
    prompt = build_single_requirement_prompt(
        method=method,
        requirement_payload=requirement_payload,
        instructions=instructions,
    )
    return parse_compliance_analysis_response(
        raw_analysis=llm_service.generate(prompt, temperature=temperature),
        method=f"{method} requirement",
        expected_count=1,
    )


def normalize_requirement_finding(
    *,
    finding: ComplianceFinding,
    retrieved_record_sections: list[dict[str, Any]],
    allowed_record_documents: set[str],
) -> ComplianceFinding:
    normalized = finding.model_copy(
        update={
            "source_documents": sanitize_source_documents(
                finding.source_documents,
                allowed=allowed_record_documents,
                fallback=[item.get("source_document") for item in retrieved_record_sections],
            ),
            "evidence": normalize_string_list(finding.evidence),
        }
    )
    return verify_finding_against_retrieved_sections(
        finding=normalized,
        retrieved_sections=retrieved_record_sections,
    )


def normalize_requirement_linked_row(
    *,
    finding: ComplianceFinding,
    row: ComplianceLinkedRow | None,
) -> ComplianceLinkedRow:
    return ComplianceLinkedRow(
        requirement_ref=row.requirement_ref if row and row.requirement_ref else "",
        requirement=row.requirement if row and row.requirement else finding.requirement,
        status=finding.status,
        gap="" if finding.status == "satisfied" else (row.gap if row and row.gap else finding.requirement),
        recommendation=(
            ""
            if finding.status == "satisfied"
            else (
                sanitize_compliance_recommendation(row.recommendation)
                if row and row.recommendation
                else recommended_action_for_requirement(finding.requirement)
            )
        ),
    )


def compute_completion_percent(findings: list[ComplianceFinding]) -> int:
    if not findings:
        return 0
    total_score = sum(STATUS_COMPLETION_SCORES.get(finding.status, 0) for finding in findings)
    return round(total_score / len(findings))


def compute_overall_assessment_from_findings(findings: list[ComplianceFinding]) -> str:
    completion_percent = compute_completion_percent(findings)
    if completion_percent <= 20:
        return "Completed_0_20"
    if completion_percent <= 40:
        return "Completed_21_40"
    if completion_percent <= 60:
        return "Completed_41_60"
    if completion_percent <= 80:
        return "Completed_61_80"
    return "Completed_81_100"


def apply_computed_overall_assessment(analysis: ComplianceAnalysis) -> ComplianceAnalysis:
    findings = analysis.procedure_to_record or analysis.findings
    completion_percent = compute_completion_percent(findings)
    return analysis.model_copy(
        update={
            "overall_assessment": compute_overall_assessment_from_findings(findings),
            "completion_percent": completion_percent,
        }
    )


def build_linked_rows_from_findings(
    findings: list[ComplianceFinding],
    existing_rows: list[ComplianceLinkedRow] | None = None,
) -> list[ComplianceLinkedRow]:
    rows = existing_rows or []
    resolved_rows: list[ComplianceLinkedRow] = []
    for index, finding in enumerate(findings):
        existing_row = rows[index] if index < len(rows) else None
        resolved_rows.append(
            ComplianceLinkedRow(
                requirement_ref=f"REQ-{index + 1}",
                requirement=(existing_row.requirement if existing_row and existing_row.requirement else finding.requirement),
                status=finding.status,
                record_recall_at_k=(existing_row.record_recall_at_k if existing_row else None),
                gap=(
                    ""
                    if finding.status == "satisfied"
                    else (
                        existing_row.gap
                        if existing_row and existing_row.gap
                        else finding.requirement
                    )
                ),
                recommendation=(
                    ""
                    if finding.status == "satisfied"
                    else (
                        sanitize_compliance_recommendation(existing_row.recommendation)
                        if existing_row and existing_row.recommendation
                        else recommended_action_for_requirement(finding.requirement)
                    )
                ),
            )
        )
    return resolved_rows


def assemble_compliance_analysis(
    *,
    findings: list[ComplianceFinding],
    linked_rows: list[ComplianceLinkedRow] | None = None,
) -> ComplianceAnalysis:
    resolved_rows = build_linked_rows_from_findings(findings, existing_rows=linked_rows)
    analysis = ComplianceAnalysis(
        overall_assessment="computed_by_backend",
        completion_percent=0,
        linked_rows=resolved_rows,
        findings=findings,
        procedure_to_record=findings,
        gaps=[row.gap for row in resolved_rows if row.status != "satisfied" and row.gap],
        recommended_actions=[
            row.recommendation
            for row in resolved_rows
            if row.status != "satisfied" and row.recommendation
        ],
    )
    return apply_computed_overall_assessment(analysis)


def recommended_action_for_requirement(requirement: str) -> str:
    text = " ".join(str(requirement or "").split())
    if not text:
        return "Address the missing requirement."

    replacements = [
        (r"^In cases where (.+?) shall record (.+)$", r"When \1, record \2."),
        (r"^(.+?) shall be documented\.?$", r"Document \1."),
        (r"^(.+?) shall be defined\.?$", r"Define \1."),
        (r"^(.+?) shall be implemented\.?$", r"Implement \1."),
        (r"^(.+?) shall be performed\.?$", r"Perform \1."),
        (r"^(.+?) shall be used\.?$", r"Use \1."),
        (r"^(.+?) shall be described\.?$", r"Describe \1."),
        (r"^(.+?) shall record (.+)$", r"Record \2 for \1."),
    ]
    for pattern, replacement in replacements:
        if re.match(pattern, text, flags=re.IGNORECASE):
            rewritten = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            return sanitize_compliance_recommendation(rewritten[0].upper() + rewritten[1:])
    return sanitize_compliance_recommendation(f"Address requirement: {text}")


def validate_analysis_cardinality(
    *,
    analysis: ComplianceAnalysis,
    expected_count: int,
    method: str,
) -> None:
    if expected_count <= 0:
        return
    actual_count = len(analysis.procedure_to_record or analysis.findings)
    if actual_count != expected_count:
        raise LLMGenerationError(
            f"{method} returned {actual_count} requirement evaluations; expected {expected_count}."
        )


def verify_finding_against_retrieved_sections(
    *,
    finding: ComplianceFinding,
    retrieved_sections: list[dict[str, Any]],
) -> ComplianceFinding:
    supported_evidence = [
        evidence
        for evidence in finding.evidence
        if evidence_supported_by_sections(evidence, retrieved_sections)
    ]
    next_status = finding.status
    next_sources = list(finding.source_documents)
    if not supported_evidence:
        next_sources = []
        if finding.status == "satisfied":
            next_status = "partial"
        elif finding.status == "partial":
            next_status = "not_satisfied"
    return finding.model_copy(
        update={
            "status": next_status,
            "evidence": supported_evidence,
            "source_documents": next_sources,
        }
    )


def evidence_supported_by_sections(evidence: str, sections: list[dict[str, Any]]) -> bool:
    normalized_evidence = normalize_whitespace(evidence).lower()
    if not normalized_evidence:
        return False
    section_text = " ".join(
        normalize_whitespace(
            " ".join(part for part in (section.get("text"), section.get("table_markdown")) if part)
        ).lower()
        for section in sections
    )
    if not section_text:
        return False
    if normalized_evidence in section_text:
        return True
    evidence_tokens = re.findall(r"[a-z0-9]+", normalized_evidence)
    if not evidence_tokens:
        return False
    section_tokens = set(re.findall(r"[a-z0-9]+", section_text))
    overlap = sum(1 for token in evidence_tokens if token in section_tokens)
    min_overlap = min(len(evidence_tokens), 4)
    return overlap >= min_overlap and (overlap / len(evidence_tokens)) >= 0.7


def normalize_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        normalized = normalize_whitespace(value)
        return [normalized] if normalized else []
    if not isinstance(value, list):
        return []
    return [normalize_whitespace(str(item)) for item in value if normalize_whitespace(str(item))]


def sanitize_source_documents(
    value: Any,
    *,
    allowed: set[str],
    fallback: list[Any],
) -> list[str]:
    requested = normalize_string_list(value)
    sanitized = [item for item in requested if item in allowed]
    if sanitized:
        return sanitized
    fallback_sanitized = [
        normalize_whitespace(str(item))
        for item in fallback
        if normalize_whitespace(str(item)) in allowed
    ]
    return list(dict.fromkeys(fallback_sanitized))


def sanitize_compliance_recommendation(value: str) -> str:
    text = normalize_whitespace(value)
    if not text:
        return ""
    replacements = [
        ("procedure deliverables", "record documentation"),
        ("procedure deliverable", "record document"),
        ("procedure documents", "record documents"),
        ("procedure document", "record document"),
        ("procedure text", "record documentation"),
    ]
    sanitized = text
    for old, new in replacements:
        sanitized = re.sub(old, new, sanitized, flags=re.IGNORECASE)
    return sanitized


def _normalize_analysis_source_documents(
    *,
    analysis: ComplianceAnalysis,
    allowed_record_documents: set[str],
) -> ComplianceAnalysis:
    normalized_procedure_to_record = [
        _normalize_finding_source_documents(
            finding=finding,
            allowed_record_documents=allowed_record_documents,
        )
        for finding in analysis.procedure_to_record
    ]
    normalized_findings = [
        _normalize_finding_source_documents(
            finding=finding,
            allowed_record_documents=allowed_record_documents,
        )
        for finding in analysis.findings
    ]
    return analysis.model_copy(
        update={
            "procedure_to_record": normalized_procedure_to_record,
            "findings": normalized_findings,
        }
    )


def _normalize_finding_source_documents(
    *,
    finding: ComplianceFinding,
    allowed_record_documents: set[str],
) -> ComplianceFinding:
    normalized_evidence = normalize_string_list(finding.evidence)
    if not normalized_evidence:
        return finding.model_copy(
            update={
                "evidence": [],
                "source_documents": [],
            }
        )
    return finding.model_copy(
        update={
            "evidence": normalized_evidence,
            "source_documents": sanitize_source_documents(
                finding.source_documents,
                allowed=allowed_record_documents,
                fallback=[],
            ),
        }
    )


def _sanitize_linked_row_recommendation(row: ComplianceLinkedRow) -> ComplianceLinkedRow:
    return row.model_copy(
        update={
            "recommendation": sanitize_compliance_recommendation(row.recommendation),
        }
    )


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
